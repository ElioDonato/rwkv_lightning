import logging

logger = logging.getLogger("state.pool")

### State Pool Manager for RWKV-7 Inference
### Manages three-level session caching plus RAM+disk prefix-state cache
import hashlib
import io
import sqlite3
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

L1_CAPACITY = 16  # VRAM (Hot)
L2_CAPACITY = 64  # RAM (Warm)
DB_PATH = "rwkv_sessions.db"  # infinite cold state pool HaHa!

PREFIX_CACHE_BUCKETS = (1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192)
PREFIX_CACHE_BUCKET_CAPACITY = 16
PREFIX_HASH_COLUMNS = tuple(f"prefix_hash_{bucket}" for bucket in PREFIX_CACHE_BUCKETS)


def _serialize_token_ids(tokens: List[int] | Tuple[int, ...]) -> str:
    return " ".join(str(token) for token in tokens)


def _deserialize_token_ids(serialized: str) -> Tuple[int, ...]:
    if not serialized:
        return ()
    return tuple(int(token) for token in serialized.split(" "))


def _hash_token_ids(tokens: List[int] | Tuple[int, ...]) -> str:
    payload = _serialize_token_ids(tokens).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _build_prefix_hashes(tokens: List[int] | Tuple[int, ...]) -> Dict[int, Optional[str]]:
    prefix_hashes: Dict[int, Optional[str]] = {}
    token_count = len(tokens)
    for bucket in PREFIX_CACHE_BUCKETS:
        prefix_hashes[bucket] = _hash_token_ids(tokens[:bucket]) if token_count >= bucket else None
    return prefix_hashes


class _CompressedTrieNode:
    def __init__(self, label: Tuple[int, ...] = ()):
        self.label: Tuple[int, ...] = label
        self.children: Dict[int, "_CompressedTrieNode"] = {}
        self.terminal_key: Optional[str] = None


class _CompressedTrie:
    def __init__(self):
        self.root = _CompressedTrieNode()

    @staticmethod
    def _common_prefix_len(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
        limit = min(len(a), len(b))
        idx = 0
        while idx < limit and a[idx] == b[idx]:
            idx += 1
        return idx

    def clear(self):
        self.root = _CompressedTrieNode()

    def insert(self, tokens: Tuple[int, ...], terminal_key: str):
        self._insert(self.root, tokens, terminal_key)

    def _insert(self, node: _CompressedTrieNode, tokens: Tuple[int, ...], terminal_key: str):
        if not tokens:
            node.terminal_key = terminal_key
            return

        first = tokens[0]
        child = node.children.get(first)
        if child is None:
            new_child = _CompressedTrieNode(tokens)
            new_child.terminal_key = terminal_key
            node.children[first] = new_child
            return

        common = self._common_prefix_len(tokens, child.label)
        if common == len(child.label):
            self._insert(child, tokens[common:], terminal_key)
            return

        split_label = child.label[:common]
        split_node = _CompressedTrieNode(split_label)
        node.children[first] = split_node

        child.label = child.label[common:]
        split_node.children[child.label[0]] = child

        remaining = tokens[common:]
        if remaining:
            new_child = _CompressedTrieNode(remaining)
            new_child.terminal_key = terminal_key
            split_node.children[remaining[0]] = new_child
        else:
            split_node.terminal_key = terminal_key

    def longest_prefix(self, tokens: List[int] | Tuple[int, ...]) -> Tuple[Optional[str], int]:
        node = self.root
        idx = 0
        best_key = node.terminal_key
        best_len = 0 if best_key is not None else 0
        tokens_tuple = tuple(tokens)

        while idx < len(tokens_tuple):
            child = node.children.get(tokens_tuple[idx])
            if child is None:
                break

            label = child.label
            if tokens_tuple[idx : idx + len(label)] != label:
                break

            idx += len(label)
            node = child
            if node.terminal_key is not None:
                best_key = node.terminal_key
                best_len = idx

        return best_key, best_len


@dataclass
class PrefixCacheEntry:
    state_id: str
    bucket_len: int
    token_count: int
    prefix_tokens: Tuple[int, ...]
    prefix_hashes: Dict[int, Optional[str]]
    state_cpu: List[torch.Tensor]
    logits_cpu: Optional[torch.Tensor]
    last_updated: float
    # Model namespace this entry belongs to ("" = the default model). Kept so a
    # shared entry index / DB can rebuild per-model tries and stay isolated when
    # multiple RWKV models run in one process.
    model: str = ""


def _session_key(model, session_id: str) -> str:
    """Scope a session key by model namespace (""/falsy = identity)."""
    if not model:
        return session_id
    return f"{model}:{session_id}"


def _prefix_key(model, raw_state_id: str) -> str:
    """Scope a prefix state_id by model namespace (""/falsy = identity)."""
    if not model:
        return raw_state_id
    return f"{model}:{raw_state_id}"


class StateCacheManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(StateCacheManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.l1_cache: OrderedDict[str, List[torch.Tensor]] = OrderedDict()
        
        self.l2_cache: OrderedDict[str, List[torch.Tensor]] = OrderedDict()
        
        self.prefix_l2_cache: Dict[int, OrderedDict[str, PrefixCacheEntry]] = {
            bucket: OrderedDict() for bucket in PREFIX_CACHE_BUCKETS
        }
        self.prefix_entry_index: Dict[str, PrefixCacheEntry] = {}
        # One compressed trie PER model namespace ("" = default model), so a
        # token prefix of one model can never resolve to another model's state.
        self.prefix_tries: Dict[str, _CompressedTrie] = {}
        self.prefix_trie = _CompressedTrie()  # back-compat alias for the default model
        self.prefix_tries[""] = self.prefix_trie
        
        self.cache_lock = threading.RLock()
        
        self.db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.db_cursor = self.db_conn.cursor()
        self.db_lock = threading.Lock()
        
        self._init_db()
        
        self.io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="db_writer")
        
        # Background DB sweeper (A3c); None until start_sweeper() is called.
        self._sweep_stop = None
        self._sweep_thread = None
        self._initialized = True
        logger.info(f"[StatePool] Initialized. L1: {L1_CAPACITY}, L2: {L2_CAPACITY}, "
            f"Prefix L2: {len(PREFIX_CACHE_BUCKETS)}x{PREFIX_CACHE_BUCKET_CAPACITY}, DB: {DB_PATH}")

    def _init_db(self):
        """初始化数据库表"""
        with self.db_lock:
            self.db_cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    state_blob BLOB,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            prefix_hash_sql = ", ".join(f"{column} TEXT" for column in PREFIX_HASH_COLUMNS)
            self.db_cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS prefix_cache (
                    state_id TEXT PRIMARY KEY,
                    bucket_len INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    {prefix_hash_sql},
                    state_blob BLOB NOT NULL,
                    logits_blob BLOB,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            for bucket in PREFIX_CACHE_BUCKETS:
                self.db_cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_prefix_cache_{bucket}
                    ON prefix_cache (bucket_len, prefix_hash_{bucket}, last_updated)
                    """
                )
            self.db_conn.commit()

    def _serialize(self, state) -> bytes:
        buffer = io.BytesIO()
        torch.save(state, buffer)
        return buffer.getvalue()

    def _deserialize(self, blob: bytes):
        buffer = io.BytesIO(blob)
        return torch.load(buffer, map_location="cpu", weights_only=True)

    def _clone_state(self, state: List[torch.Tensor]) -> List[torch.Tensor]:
        """深拷贝状态，避免多线程共享导致污染"""
        return [t.clone() for t in state]

    def _clone_to_cpu_state(self, state: List[torch.Tensor]) -> List[torch.Tensor]:
        return [t.detach().to("cpu").clone() for t in state]

    def _clone_to_device_state(self, state: List[torch.Tensor], device: str) -> List[torch.Tensor]:
        return [t.detach().to(device, non_blocking=device == "cuda").clone() for t in state]

    def _clone_optional_tensor(self, tensor: Optional[torch.Tensor], device: str) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor.detach().to(device, non_blocking=device == "cuda").clone()

    def _persist_task(self, session_id: str, state_cpu: List[torch.Tensor]):
        """异步任务：序列化并写入数据库"""
        try:
            blob = self._serialize(state_cpu)
            with self.db_lock:
                self.db_cursor.execute(
                    "INSERT OR REPLACE INTO sessions (session_id, state_blob, last_updated) VALUES (?, ?, ?)",
                    (session_id, blob, time.time())
                )
                self.db_conn.commit()
            # print(f"[StatePool] Persisted session {session_id} to L3 (Disk).")
            
            # 显式删除引用协助 GC
            del state_cpu
            del blob
        except Exception as e:
            logger.error(f"[StatePool] Error persisting session {session_id}: {e}")

    def _persist_prefix_task(self, entry: PrefixCacheEntry):
        try:
            state_blob = self._serialize(entry.state_cpu)
            logits_blob = self._serialize(entry.logits_cpu) if entry.logits_cpu is not None else None
            with self.db_lock:
                row = [
                    entry.state_id,
                    entry.bucket_len,
                    entry.token_count,
                ]
                row.extend(entry.prefix_hashes.get(bucket) for bucket in PREFIX_CACHE_BUCKETS)
                row.extend([state_blob, logits_blob, entry.last_updated])

                placeholders = ", ".join("?" for _ in row)
                columns = ", ".join(
                    ["state_id", "bucket_len", "token_count", *PREFIX_HASH_COLUMNS, "state_blob", "logits_blob", "last_updated"]
                )
                self.db_cursor.execute(
                    f"INSERT OR REPLACE INTO prefix_cache ({columns}) VALUES ({placeholders})",
                    row,
                )
                self.db_conn.commit()
        except Exception as e:
            logger.error(f"[StatePool] Error persisting prefix cache {entry.state_id[:96]}...: {e}")

    def _rebuild_prefix_trie(self):
        """Rebuild one compressed trie per model namespace from the shared
        entry index, so a token prefix of one model never resolves to another
        model's recurrent state (airtight even for same-vocab checkpoints)."""
        models = {e.model for e in self.prefix_entry_index.values()} or {""}
        self.prefix_tries = {m: _CompressedTrie() for m in models}
        self.prefix_trie = self.prefix_tries.setdefault("", _CompressedTrie())
        for entry in self.prefix_entry_index.values():
            self.prefix_tries.setdefault(entry.model, _CompressedTrie()).insert(
                entry.prefix_tokens, entry.state_id
            )

    def _store_prefix_entry_locked(self, entry: PrefixCacheEntry, persist: bool):
        bucket_cache = self.prefix_l2_cache.setdefault(entry.bucket_len, OrderedDict())
        if entry.state_id in bucket_cache:
            del bucket_cache[entry.state_id]
        bucket_cache[entry.state_id] = entry
        self.prefix_entry_index[entry.state_id] = entry

        evicted_entry = None
        if len(bucket_cache) > PREFIX_CACHE_BUCKET_CAPACITY:
            _, evicted_entry = bucket_cache.popitem(last=False)
            self.prefix_entry_index.pop(evicted_entry.state_id, None)

        # Trie update. A plain INSERT is an O(1)-amortized child-insert shared by
        # both put_prefix_state and the disk-load path, keeping the hot path
        # incremental instead of an O(n) full rebuild every insert. An EVICTION
        # forces a full rebuild immediately: the radix trie has no point-delete,
        # and leaving a stale (evicted) terminal could otherwise SHADOW a shorter
        # live terminal and change hit/miss on the default-ON prefix path vs the
        # old always-rebuild behavior -- a byte-identity violation. Rebuilding on
        # eviction (rare, only on bucket overflow) keeps the trie always exactly
        # equal to the live index, identical to the pre-A3a behavior.
        if evicted_entry is not None:
            self._rebuild_prefix_trie()
        else:
            trie = self.prefix_tries.setdefault(entry.model, _CompressedTrie())
            trie.insert(entry.prefix_tokens, entry.state_id)
            if entry.model == "":
                # keep the back-compat alias pointing at the default model's trie
                self.prefix_trie = trie

        if persist and entry.bucket_len in PREFIX_CACHE_BUCKETS:
            self._submit_persist(self._persist_prefix_task, entry)
        # An ADAPTIVE-length entry (see put_prefix_state) is L2-only: it has no
        # matching fixed prefix_hash_<bucket> column to be persisted under, and a
        # persisted NULL-hash row would be unreachable+wasteful -- so it is never
        # persisted, on insert OR on eviction (documented retention tradeoff:
        # adaptive checkpoints are lost on restart / after L2 eviction).
        if evicted_entry is not None and evicted_entry.bucket_len in PREFIX_CACHE_BUCKETS:
            self._submit_persist(self._persist_prefix_task, evicted_entry)
        # Prune an ADAPTIVE bucket key that just became empty, so the number of
        # distinct-length bucket keys (which adaptive checkpoints create) doesn't
        # grow without bound as requests come and go with different prompt
        # lengths. Fixed-bucket keys are always present by construction.
        if evicted_entry is not None and evicted_entry.bucket_len not in PREFIX_CACHE_BUCKETS:
            if not bucket_cache:
                self.prefix_l2_cache.pop(evicted_entry.bucket_len, None)

    def _submit_persist(self, task, *args):
        """Submit a persist task off the event loop, tolerating a shutdown race
        (io_executor already shut down by flush_all) so a late in-flight request
        never crashes on ``cannot schedule new futures after shutdown``."""
        try:
            self.io_executor.submit(task, *args)
        except RuntimeError:
            pass  # executing during/after shutdown -- persist is best-effort

    def put_state(self, session_id: str, state: List[torch.Tensor], model=None):
        """
        存入状态。
        流程：
        1. 存入 L1 (GPU)。
        2. 如果 L1 满 -> 移出最久未使用的到 L2 (CPU)。
        3. 如果 L2 满 -> 移出最久未使用的到 L3 (Disk, Async)。
        """
        if session_id is None:
            return
        session_id = _session_key(model, session_id)

        with self.cache_lock:
            if session_id in self.l1_cache:
                del self.l1_cache[session_id]
            if session_id in self.l2_cache:
                del self.l2_cache[session_id]
            
            self.l1_cache[session_id] = state
            
            if len(self.l1_cache) > L1_CAPACITY:
                # popitem(last=False) 弹出最早插入的元素 (FIFO/LRU Oldest)
                evicted_id, evicted_state_gpu = self.l1_cache.popitem(last=False)
                
                evicted_state_cpu = [t.to('cpu', non_blocking=True) for t in evicted_state_gpu]
                
                self.l2_cache[evicted_id] = evicted_state_cpu
                
                if len(self.l2_cache) > L2_CAPACITY:
                    l2_evicted_id, l2_evicted_state_cpu = self.l2_cache.popitem(last=False)

                    self._submit_persist(self._persist_task, l2_evicted_id, l2_evicted_state_cpu)

    def get_state(self, session_id: str, model=None) -> Optional[List[torch.Tensor]]:

        if session_id is None:
            return None
        session_id = _session_key(model, session_id)

        with self.cache_lock:
            # Case 1: L1 Hit (VRAM)
            if session_id in self.l1_cache:
                self.l1_cache.move_to_end(session_id) # 标记为最近使用
                state = self.l1_cache[session_id]
                token_pos = state[2].item() if len(state) > 2 and hasattr(state[2], "item") else "unknown"
                logger.info(f"[StatePool][SESSION HIT][L1] session_id={session_id} "
                    f"token_pos={token_pos}")
                return self._clone_state(self.l1_cache[session_id])
            
            # Case 2: L2 Hit (RAM)
            if session_id in self.l2_cache:
                state_cpu = self.l2_cache.pop(session_id)
                token_pos = state_cpu[2].item() if len(state_cpu) > 2 and hasattr(state_cpu[2], "item") else "unknown"
                logger.info(f"[StatePool][SESSION HIT][L2] session_id={session_id} "
                    f"token_pos={token_pos} -> promote_to_l1")
                state_gpu = [t.to('cuda', non_blocking=True) for t in state_cpu]
                
                self.put_state(session_id, state_gpu)
                return self._clone_state(state_gpu)

        blob = None
        with self.db_lock:
            self.db_cursor.execute("SELECT state_blob FROM sessions WHERE session_id = ?", (session_id,))
            row = self.db_cursor.fetchone()
            if row:
                blob = row[0]
        
        if blob:
            try:
                state_cpu = self._deserialize(blob)
                token_pos = state_cpu[2].item() if len(state_cpu) > 2 and hasattr(state_cpu[2], "item") else "unknown"
                logger.info(f"[StatePool][SESSION HIT][DISK] session_id={session_id} "
                    f"token_pos={token_pos} -> load_to_l1")
                state_gpu = [t.to('cuda') for t in state_cpu]
                self.put_state(session_id, state_gpu)
                return self._clone_state(state_gpu)
            except Exception as e:
                logger.warning(f"[StatePool] Failed to deserialize session {session_id}: {e}")
                return None

        return None

    def has_state(self, session_id: str, model=None) -> bool:
        if session_id is None:
            return False
        session_id = _session_key(model, session_id)

        with self.cache_lock:
            if session_id in self.l1_cache or session_id in self.l2_cache:
                return True

        with self.db_lock:
            self.db_cursor.execute(
                "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1", (session_id,)
            )
            return self.db_cursor.fetchone() is not None

    def put_prefix_state(
        self,
        prefix_tokens: List[int] | Tuple[int, ...],
        state: List[torch.Tensor],
        logits: Optional[torch.Tensor] = None,
        model=None,
    ) -> bool:
        token_tuple = tuple(prefix_tokens)
        bucket_len = len(token_tuple)
        # An empty prefix would create a ROOT terminal in the trie, making every
        # later match (even one sharing no tokens) resolve to it at matched_len 0
        # -- a silent wrong-state serve. Never store empty checkpoints.
        if bucket_len == 0:
            return False
        fixed = bucket_len in PREFIX_CACHE_BUCKETS
        if not fixed and not self._prefix_adaptive_enabled():
            return False

        entry = PrefixCacheEntry(
            state_id=_prefix_key(model, _serialize_token_ids(token_tuple)),
            bucket_len=bucket_len,
            token_count=bucket_len,
            prefix_tokens=token_tuple,
            prefix_hashes=_build_prefix_hashes(token_tuple),
            state_cpu=self._clone_to_cpu_state(state),
            logits_cpu=self._clone_optional_tensor(logits, "cpu"),
            last_updated=time.time(),
            model=model or "",
        )
        # Fixed buckets persist to SQLite as before. Adaptive lengths (B,
        # RWKV_PREFIX_ADAPTIVE) are L2-only -- L2/trie matches recover them for
        # the next request, but they have no fixed hash column to be indexed by
        # on disk and are dropped on eviction/restart (retention tradeoff).
        with self.cache_lock:
            self._store_prefix_entry_locked(entry, persist=fixed)
        return True

    @staticmethod
    def _prefix_adaptive_enabled() -> bool:
        """Lazy settings read (same pattern as _prefix_disk_async_enabled) so
        the state core stays importable without the server settings module.
        RWKV_TURN_STATE_REUSE is an alias that enables the same adaptive
        short-prompt / multi-turn prefix reuse on the raw chat paths."""
        try:
            from settings import settings as _settings
            return bool(
                getattr(_settings, "prefix_adaptive", False)
                or getattr(_settings, "turn_state_reuse", False)
            )
        except Exception:
            return False

    def _load_prefix_entry_from_db_locked(
        self,
        prefix_tokens: List[int] | Tuple[int, ...],
        bucket_len: int,
        model=None,
    ) -> Optional[PrefixCacheEntry]:
        state_id = _prefix_key(model, _serialize_token_ids(prefix_tokens[:bucket_len]))
        hash_column = f"prefix_hash_{bucket_len}"
        hash_value = _hash_token_ids(prefix_tokens[:bucket_len])

        with self.db_lock:
            self.db_cursor.execute(
                f"""
                SELECT state_blob, logits_blob, last_updated
                FROM prefix_cache
                WHERE state_id = ? AND bucket_len = ? AND {hash_column} = ?
                LIMIT 1
                """,
                (state_id, bucket_len, hash_value),
            )
            row = self.db_cursor.fetchone()

        if row is None:
            return None

        try:
            state_cpu = self._deserialize(row[0])
            logits_cpu = self._deserialize(row[1]) if row[1] is not None else None
        except Exception as e:
            logger.warning(f"[StatePool] Failed to deserialize prefix cache {state_id[:96]}...: {e}")
            return None

        entry = PrefixCacheEntry(
            state_id=state_id,
            bucket_len=bucket_len,
            token_count=bucket_len,
            prefix_tokens=tuple(prefix_tokens[:bucket_len]),
            prefix_hashes=_build_prefix_hashes(prefix_tokens[:bucket_len]),
            state_cpu=state_cpu,
            logits_cpu=logits_cpu,
            last_updated=float(row[2]) if row[2] is not None else time.time(),
            model=model or "",
        )
        self._store_prefix_entry_locked(entry, persist=False)
        return entry

    def match_prefix_state(
        self,
        prompt_tokens: List[int] | Tuple[int, ...],
        device: str = "cuda",
        model=None,
    ) -> Optional[dict]:
        token_tuple = tuple(prompt_tokens)
        if not token_tuple:
            return None

        with self.cache_lock:
            trie = self.prefix_tries.get(model or "", self.prefix_trie)
            state_id, matched_len = trie.longest_prefix(token_tuple)
            if state_id is not None:
                entry = self.prefix_entry_index.get(state_id)
                if entry is not None:
                    bucket_cache = self.prefix_l2_cache[entry.bucket_len]
                    if state_id in bucket_cache:
                        bucket_cache.move_to_end(state_id)
                    prompt_prefix_hashes = _build_prefix_hashes(token_tuple)
                    logger.info("[StatePool][PREFIX HIT][L2] "
                        f"matched_tokens={matched_len} "
                        f"bucket_len={entry.bucket_len} "
                        f"prompt_len={len(token_tuple)} "
                        f"state_id={entry.state_id[:160]} "
                        f"hash_{entry.bucket_len}={prompt_prefix_hashes.get(entry.bucket_len)}")
                    return {
                        "state_id": entry.state_id,
                        "matched_tokens": matched_len,
                        "bucket_len": entry.bucket_len,
                        "state": self._clone_to_device_state(entry.state_cpu, device),
                        "logits": self._clone_optional_tensor(entry.logits_cpu, device),
                        "cache_source": "l2_ram",
                    }

        # Disk fallback. Default (RWKV_PREFIX_DISK_ASYNC off) keeps the
        # historical loop below: up to 8 synchronous SQLite SELECTs on the
        # request thread, byte-identical. With the knob ON, replace that with a
        # single bounded probe + background warm (see _bounded_prefix_disk_probe).
        if self._prefix_disk_async_enabled():
            return self._bounded_prefix_disk_probe(token_tuple, device, model)

        for bucket in reversed(PREFIX_CACHE_BUCKETS):
            if len(token_tuple) < bucket:
                continue
            with self.cache_lock:
                entry = self._load_prefix_entry_from_db_locked(token_tuple, bucket, model)
                if entry is not None:
                    prompt_prefix_hashes = _build_prefix_hashes(token_tuple)
                    logger.info("[StatePool][PREFIX HIT][DISK] "
                        f"matched_tokens={bucket} "
                        f"bucket_len={entry.bucket_len} "
                        f"prompt_len={len(token_tuple)} "
                        f"state_id={entry.state_id[:160]} "
                        f"hash_{entry.bucket_len}={prompt_prefix_hashes.get(entry.bucket_len)} "
                        "-> load_to_l2")
                    return {
                        "state_id": entry.state_id,
                        "matched_tokens": bucket,
                        "bucket_len": entry.bucket_len,
                        "state": self._clone_to_device_state(entry.state_cpu, device),
                        "logits": self._clone_optional_tensor(entry.logits_cpu, device),
                        "cache_source": "disk",
                    }

        return None

    @staticmethod
    def _prefix_disk_async_enabled() -> bool:
        """Lazy settings read so the state core stays importable without the
        server settings module; a missing import (isolated harness) means the
        historical (off) behavior is used."""
        try:
            from settings import settings as _settings
            return bool(getattr(_settings, "prefix_disk_async", False))
        except Exception:
            return False

    def _bounded_prefix_disk_probe(
        self,
        token_tuple: Tuple[int, ...],
        device: str,
        model: Optional[str],
    ) -> Optional[dict]:
        """A3b (RWKV_PREFIX_DISK_ASYNC): replace the up-to-8 synchronous disk
        probes with ONE bounded read at the largest plausible bucket <= prompt
        length, then warm any smaller matching bucket in the background so the
        NEXT request finds it in L2 (off the event loop). Recognized reduced
        recall vs the full loop for a single request (only the largest bucket is
        probed synchronously; smaller matches surface one request later via the
        warm) -- a documented throughput-vs-recall tradeoff, opt-in only."""
        candidate = None
        for bucket in reversed(PREFIX_CACHE_BUCKETS):
            if len(token_tuple) >= bucket:
                candidate = bucket
                break
        if candidate is None:
            return None

        with self.cache_lock:
            entry = self._load_prefix_entry_from_db_locked(token_tuple, candidate, model)
        if entry is not None:
            prompt_prefix_hashes = _build_prefix_hashes(token_tuple)
            logger.info("[StatePool][PREFIX HIT][DISK] "
                f"matched_tokens={candidate} "
                f"bucket_len={entry.bucket_len} "
                f"prompt_len={len(token_tuple)} "
                f"state_id={entry.state_id[:160]} "
                f"hash_{entry.bucket_len}={prompt_prefix_hashes.get(entry.bucket_len)} "
                "-> load_to_l2")
            return {
                "state_id": entry.state_id,
                "matched_tokens": candidate,
                "bucket_len": entry.bucket_len,
                "state": self._clone_to_device_state(entry.state_cpu, device),
                "logits": self._clone_optional_tensor(entry.logits_cpu, device),
                "cache_source": "disk",
            }

        def _warm_smaller_buckets():
            # Recheck the candidate too: an entry may have been evicted from
            # L2 while this warm was queued.
            with self.cache_lock:
                for bucket in sorted(PREFIX_CACHE_BUCKETS, reverse=True):
                    if bucket >= candidate or len(token_tuple) < bucket:
                        continue
                    self._load_prefix_entry_from_db_locked(token_tuple, bucket, model)

        try:
            self.io_executor.submit(_warm_smaller_buckets)
        except RuntimeError:
            pass  # io_executor already shut down -> nothing to warm

        return None

    def close_session(self, session_id: str, model=None):

        state_to_save = None
        session_id = _session_key(model, session_id)
        
        with self.cache_lock:
            if session_id in self.l1_cache:
                state_to_save = [t.to('cpu') for t in self.l1_cache.pop(session_id)]
            elif session_id in self.l2_cache:
                state_to_save = self.l2_cache.pop(session_id)
        
        if state_to_save:
            self._persist_task(session_id, state_to_save)
        
        logger.info(f"[StatePool] Session {session_id} closed and persisted.")

    def flush_all(self):

        self.stop_sweeper()
        logger.info("[StatePool] Flushing all states to disk...")
        
        self.io_executor.shutdown(wait=True)
        
        items_to_save = []
        prefix_entries_to_save: List[PrefixCacheEntry] = []
        with self.cache_lock:
            while self.l1_cache:
                sid, state = self.l1_cache.popitem()
                items_to_save.append((sid, [t.to('cpu') for t in state]))
            
            while self.l2_cache:
                sid, state = self.l2_cache.popitem()
                items_to_save.append((sid, state))

            for bucket_cache in self.prefix_l2_cache.values():
                while bucket_cache:
                    _, entry = bucket_cache.popitem()
                    # Adaptive (L2-only) entries are never persisted: they have
                    # no fixed prefix_hash_<bucket> column to be indexed by, and a
                    # NULL-hash row would be unreachable + later re-appear as
                    # dead weight under a sweep cap. Only fixed-bucket rows flush.
                    if entry.bucket_len in PREFIX_CACHE_BUCKETS:
                        prefix_entries_to_save.append(entry)
            self.prefix_entry_index.clear()
            self._rebuild_prefix_trie()

        with self.db_lock:
            try:
                self.db_conn.execute("BEGIN TRANSACTION")
                for sid, state in items_to_save:
                    blob = self._serialize(state)
                    self.db_conn.execute(
                        "INSERT OR REPLACE INTO sessions (session_id, state_blob, last_updated) VALUES (?, ?, ?)",
                        (sid, blob, time.time()),
                    )

                for entry in prefix_entries_to_save:
                    state_blob = self._serialize(entry.state_cpu)
                    logits_blob = self._serialize(entry.logits_cpu) if entry.logits_cpu is not None else None
                    row = [
                        entry.state_id,
                        entry.bucket_len,
                        entry.token_count,
                    ]
                    row.extend(entry.prefix_hashes.get(bucket) for bucket in PREFIX_CACHE_BUCKETS)
                    row.extend([state_blob, logits_blob, entry.last_updated])
                    placeholders = ", ".join("?" for _ in row)
                    columns = ", ".join(
                        ["state_id", "bucket_len", "token_count", *PREFIX_HASH_COLUMNS, "state_blob", "logits_blob", "last_updated"]
                    )
                    self.db_conn.execute(
                        f"INSERT OR REPLACE INTO prefix_cache ({columns}) VALUES ({placeholders})",
                        row,
                    )

                self.db_conn.commit()
                logger.info(f"[StatePool] Successfully saved {len(items_to_save)} sessions "
                    f"and {len(prefix_entries_to_save)} prefix states.")
            except Exception as e:
                logger.error(f"[StatePool] Error during flush: {e}")
                self.db_conn.rollback()
            finally:
                self.db_conn.close()

    def list_states_in_db(self) -> List[Tuple[str, float]]:

        with self.db_lock:
            self.db_cursor.execute("SELECT session_id, last_updated FROM sessions ORDER BY last_updated DESC")
            results = self.db_cursor.fetchall()
            return [(row[0], row[1]) for row in results]

    def run_sweep(self, ttl_s: float = 0.0, max_rows: int = 0) -> Dict[str, int]:
        """Prune the state DB: delete expired (older than ``ttl_s``) and, when
        ``max_rows`` > 0, over-cap rows from ``sessions`` / ``prefix_cache``,
        then ``VACUUM`` to reclaim the on-disk bloat that years of bare
        ``INSERT OR REPLACE`` (without any removal) accumulate on a long-running
        box. Runs entirely under ``db_lock``; a caller should run it on a
        background thread so it never blocks the asyncio event loop (a slow
        ``VACUUM`` on this DB is exactly the documented 20s-stall hazard).
        Returns a summary dict. Pure no-op when ttl<=0 and max_rows<=0."""
        detail = {"expired_sessions": 0, "expired_prefix": 0, "over_cap_prefix": 0}
        if not (ttl_s and ttl_s > 0) and not (max_rows and max_rows > 0):
            return detail
        now = time.time()
        with self.db_lock:
            try:
                # A range DELETE on `last_updated` without a leading index scans
                # the WHOLE table -- on a long-running, bloated rwkv_sessions.db
                # that stalls every db_lock user (the request-path disk reads) for
                # the duration, recreating the exact event-loop stall this sweep
                # exists to fix. Build the index lazily here (off the event loop,
                # only when a sweep actually runs) so the DELETE is O(index).
                self.db_conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_last_updated "
                    "ON sessions(last_updated)")
                self.db_conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_prefix_last_updated "
                    "ON prefix_cache(last_updated)")
                self.db_conn.commit()
                if ttl_s and ttl_s > 0:
                    cutoff = now - ttl_s
                    self.db_cursor.execute(
                        "DELETE FROM sessions WHERE last_updated < ?", (cutoff,))
                    detail["expired_sessions"] = self.db_cursor.rowcount
                    self.db_cursor.execute(
                        "DELETE FROM prefix_cache WHERE last_updated < ?", (cutoff,))
                    detail["expired_prefix"] = self.db_cursor.rowcount
                if max_rows and max_rows > 0:
                    self.db_cursor.execute("SELECT COUNT(*) FROM prefix_cache")
                    if self.db_cursor.fetchone()[0] > max_rows:
                        # keep the most-recent max_rows rows, evict the rest
                        self.db_cursor.execute(
                            "DELETE FROM prefix_cache WHERE state_id IN ("
                            " SELECT state_id FROM prefix_cache"
                            " ORDER BY last_updated DESC LIMIT -1 OFFSET ?)",
                            (int(max_rows),))
                        detail["over_cap_prefix"] = self.db_cursor.rowcount
                self.db_conn.commit()
            except Exception as e:
                logger.error(f"[StatePool] Sweep error: {e}")
                self.db_conn.rollback()
                return detail
        # VACUUM runs on a DEDICATED short-lived connection, outside db_lock, so a
        # multi-second VACUUM on a large DB never holds the shared cursor / db_lock
        # that the request path (session / prefix disk reads on the event loop)
        # blocks on -- otherwise this "background" sweep would recreate the very
        # event-loop stall it is meant to remove.
        if detail["expired_sessions"] or detail["expired_prefix"] or detail["over_cap_prefix"]:
            try:
                import sqlite3 as _sq
                _vac = _sq.connect(DB_PATH)
                with _vac:
                    _vac.execute("VACUUM")
                _vac.close()
            except Exception as e:
                logger.error(f"[StatePool] VACUUM failed (skipped; DB still valid): {e}")
            logger.info(f"[StatePool] Sweep removed sessions={detail['expired_sessions']} "
                        f"prefix={detail['expired_prefix']} over_cap={detail['over_cap_prefix']}")
        return detail

    def start_sweeper(self, interval_s: float, ttl_s: float = 0.0, max_rows: int = 0):
        """Start a background daemon thread that runs :meth:`run_sweep` every
        ``interval_s``. Caller (the app lifespan, gated on RWKV_CACHE_SWEEP==1)
        must call :meth:`stop_sweeper` at shutdown so the thread joins before
        ``flush_all`` closes the connection. Runs independent of the db_writer
        thread (SQL is serialized against it by ``db_lock``)."""
        if getattr(self, "_sweep_thread", None) is not None:
            return self  # idempotent: never leak a second sweeper thread
        self._sweep_stop = threading.Event()
        self._sweep_interval = float(interval_s)
        self._sweep_ttl = float(ttl_s)
        self._sweep_max_rows = int(max_rows)

        def _loop():
            while not self._sweep_stop.wait(self._sweep_interval):
                try:
                    self.run_sweep(self._sweep_ttl, self._sweep_max_rows)
                except Exception as e:  # pragma: no cover - defensive
                    logger.error(f"[StatePool] sweeper iteration failed: {e}")

        self._sweep_thread = threading.Thread(
            target=_loop, name="state_db_sweeper", daemon=True,
        )
        self._sweep_thread.start()
        logger.info(f"[StatePool] sweeper started: interval={interval_s}s "
                    f"ttl={ttl_s}s max_rows={max_rows}")
        return self

    def stop_sweeper(self):
        """Signal the sweeper thread to exit and join it (bounded by its
        current interval, so a long-polling thread can't stall shutdown)."""
        stop = getattr(self, "_sweep_stop", None)
        thread = getattr(self, "_sweep_thread", None)
        if stop is not None:
            stop.set()
        if thread is not None:
            # If run_sweep is mid-execution (holding db_lock / running VACUUM)
            # the join blocks until it finishes -- the caller must not hold
            # db_lock when it calls this (flush_all does not). Give a long
            # VACUUM time to drain rather than racing the connection close.
            thread.join(timeout=max(getattr(self, "_sweep_interval", 0.0), 60.0))
            if thread.is_alive():
                logger.warning("[StatePool] sweeper thread did not exit in time; "
                               "draining after shutdown instead")
        self._sweep_stop = None
        self._sweep_thread = None

    def list_prefix_states_in_db(self) -> List[Tuple[str, int, float]]:
        """Full `ORDER BY last_updated` scan+sort over the entire prefix_cache
        table. On a long-running deployment this table can be far larger
        (row count and on-disk bloat, especially without periodic VACUUM)
        than the sessions table, and this query has no supporting index --
        see list_all_states()'s include_prefix_db docstring. Prefer not
        calling this on a request's hot path; if a caller needs it, consider
        running it via run_in_threadpool so a slow scan doesn't block the
        asyncio event loop (and every other in-flight request) for the
        duration.
        """
        with self.db_lock:
            self.db_cursor.execute(
                "SELECT state_id, bucket_len, last_updated FROM prefix_cache ORDER BY last_updated DESC"
            )
            results = self.db_cursor.fetchall()
            return [(row[0], int(row[1]), row[2]) for row in results]

    def list_all_states(self, include_prefix_db: bool = False) -> dict:
        """List known session-state keys across L1/L2/disk.

        `include_prefix_db` also runs list_prefix_states_in_db(), a full
        `SELECT ... FROM prefix_cache ORDER BY last_updated` scan+sort. On a
        long-running server the prefix_cache table accumulates far more rows
        (and, from years of INSERT OR REPLACE without VACUUM, far more
        on-disk bloat) than the sessions table ever does, so that scan can
        take vastly longer than everything else in this function combined --
        on this deployment's DB it hung for 20+ seconds and stalled every
        other request behind it (SQLite access here is synchronous and runs
        on the asyncio event loop thread, so a slow query blocks the whole
        server, not just its caller). Every current caller
        (collect_session_indices for /multi_state/'s dialogue_idx
        allocation, /state/delete's delete_prefix cleanup, /state/status)
        only reads l1_cache/l2_cache/database, never prefix_l2_counts,
        prefix_l2_cache, or prefix_database_count -- so default this off and
        make it opt-in for any future caller that actually needs prefix-DB
        visibility.
        """
        with self.cache_lock:
            l1_states = list(self.l1_cache.keys())
            l2_states = list(self.l2_cache.keys())
            prefix_l2_counts = {
                str(bucket): len(cache) for bucket, cache in self.prefix_l2_cache.items()
            }
            prefix_l2_keys = {
                str(bucket): list(cache.keys()) for bucket, cache in self.prefix_l2_cache.items()
            }

        db_states = self.list_states_in_db()
        db_states_keys = [state[0] for state in db_states]
        prefix_db_states = self.list_prefix_states_in_db() if include_prefix_db else []

        return {
            "l1_cache": l1_states,
            "l2_cache": l2_states,
            "database": db_states_keys,
            "total_count": len(l1_states) + len(l2_states) + len(db_states_keys),
            "prefix_l2_counts": prefix_l2_counts,
            "prefix_l2_cache": prefix_l2_keys,
            "prefix_database_count": len(prefix_db_states) if include_prefix_db else None,
        }

    def print_all_states_status(self):

        all_states = self.list_all_states()

        logger.info(f"[StatePool] All States Status - Total {all_states['total_count']} sessions:")
        logger.info("=" * 80)

        logger.info(f"L1 Cache (VRAM) - Count: {len(all_states['l1_cache'])}")
        logger.info("-" * 40)
        for session_id in all_states["l1_cache"]:
            logger.info(f"  {session_id}")

        logger.info(f"\nL2 Cache (RAM) - Count: {len(all_states['l2_cache'])}")
        logger.info("-" * 40)
        for session_id in all_states["l2_cache"]:
            logger.info(f"  {session_id}")

        logger.info(f"\nDatabase (Disk) - Count: {len(all_states['database'])}")
        logger.info("-" * 40)
        for session_id in all_states["database"]:
            logger.info(f"  {session_id}")

        logger.info("\nPrefix Cache (RAM / L2)")
        logger.info("-" * 40)
        for bucket in PREFIX_CACHE_BUCKETS:
            count = all_states["prefix_l2_counts"][str(bucket)]
            logger.info(f"  bucket={bucket}: {count}/{PREFIX_CACHE_BUCKET_CAPACITY}")

        logger.info(f"\nPrefix Cache (Disk) - Count: {all_states['prefix_database_count']}")

        if all_states["total_count"] == 0 and all_states["prefix_database_count"] == 0:
            logger.info("No sessions found in any cache level.")
        logger.info("=" * 80)

    def delete_state_from_any_level(self, session_id: str, model=None) -> bool:

        deleted_from_cache = False
        session_id = _session_key(model, session_id)

        with self.cache_lock:
            # 从L1缓存删除
            if session_id in self.l1_cache:
                del self.l1_cache[session_id]
                deleted_from_cache = True
                logger.info(f"[StatePool] Session {session_id} removed from L1 cache (VRAM).")

            # 从L2缓存删除
            if session_id in self.l2_cache:
                del self.l2_cache[session_id]
                deleted_from_cache = True
                logger.info(f"[StatePool] Session {session_id} removed from L2 cache (RAM).")

        # 从数据库删除
        with self.db_lock:
            try:
                self.db_cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                self.db_conn.commit()

                affected_rows = self.db_cursor.rowcount
                if affected_rows > 0:
                    logger.info(f"[StatePool] Session {session_id} removed from database (Disk).")
                    return True
                if deleted_from_cache:
                    return True
                logger.info(f"[StatePool] Session {session_id} not found in any cache level.")
                return False
            except Exception as e:
                logger.error(f"[StatePool] Error deleting session {session_id} from database: {e}")
                return False

def show_all_states_status():
    manager = get_state_manager()
    manager.print_all_states_status()

def remove_session_from_any_level(session_id: str) -> bool:
    manager = get_state_manager()
    return manager.delete_state_from_any_level(session_id)

def get_state_manager(model=None) -> StateCacheManager:
    """Return the process-wide StateCacheManager. When ``model`` (a model
    namespace from ``model_namespace(slot)``) is provided, return a stateless
    ``_ModelScopedManager`` that forwards it, so session + prefix state is
    isolated per model. When ``model`` is falsy/None (the default/single-model
    case) the bare singleton is returned -- byte-identical to before."""
    inst = StateCacheManager()
    if not model:
        return inst
    return _ModelScopedManager(inst, model)


class _ModelScopedManager:
    """Stateless view of the shared StateCacheManager restricted to one model
    namespace. Only the model-scoped methods forward ``model``; everything else
    delegates to the singleton."""

    _SCOPED = (
        "put_state", "get_state", "has_state", "close_session",
        "delete_state_from_any_level", "put_prefix_state", "match_prefix_state",
    )

    def __init__(self, manager: StateCacheManager, model: str):
        self._manager = manager
        self._model = model

    def put_state(self, session_id, state):
        return self._manager.put_state(session_id, state, self._model)

    def get_state(self, session_id):
        return self._manager.get_state(session_id, self._model)

    def has_state(self, session_id):
        return self._manager.has_state(session_id, self._model)

    def close_session(self, session_id):
        return self._manager.close_session(session_id, self._model)

    def delete_state_from_any_level(self, session_id):
        return self._manager.delete_state_from_any_level(session_id, self._model)

    def put_prefix_state(self, prefix_tokens, state, logits=None):
        return self._manager.put_prefix_state(prefix_tokens, state, logits, self._model)

    def match_prefix_state(self, prompt_tokens, device="cuda"):
        return self._manager.match_prefix_state(prompt_tokens, device, self._model)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._manager, name)

    def __repr__(self):
        return f"<_ModelScopedManager(model={self._model!r}, manager={self._manager!r})>"

def shutdown_state_manager():
    manager = get_state_manager()
    manager.flush_all()
