#!/usr/bin/env python3
"""Real-server throughput/latency harness for the Phase-4 gate.

Measures a live RWKV Lightning server under N concurrent streaming chat clients:
aggregate tokens/s, solo p50 TTFT and p50 per-token ms. "token" is approximated
as one streaming SSE data-frame carrying a content delta (one decode step per
token). Run against a server started with everything default-off to get the
Phase-4 baseline, then repeat after the dynamic-batch opt-in is enabled.

Usage:
  .venv/bin/python test/bench_throughput.py --url http://127.0.0.1:PORT \
      --concurrency 8 --requests 2 --max-tokens 256 [--prompt "..." ]
"""
import argparse
import asyncio
import time

import httpx

_PROMPT = ("This is an automated benchmark request. Please write a short "
           "paragraph about how transformers process sequences.")


def _count_tokens_from_stream(events):
    """Approximate tokens for a stream as the number of content-bearing frames."""
    n = 0
    for ev in events:  # ev is the 'data:' payload string
        if ev.strip() == "[DONE]":
            continue
        if '"content"' in ev and '"delta"' in ev:
            n += 1
    return max(0, n)


async def _one_client(url, prompt, max_tokens, password, results, lock, idx):
    """Run sequential requests from one client; record per-request metrics."""
    headers = {"Authorization": f"Bearer {password}"} if password else {}
    req = {
        "model": "rwkv7",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as c:
        while True:
            async with lock:
                want_more = results["pending"] > 0
                if not want_more:
                    break
                results["pending"] -= 1
            t_start = time.perf_counter()
            ttft = None
            prev = None
            frames = 0
            async with c.stream("POST", url + "/openai/v1/chat/completions",
                                json=req, headers=headers) as resp:
                if resp.status_code != 200:
                    results["errors"] += 1
                    continue
                events = []
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = (now - t_start) * 1000.0
                    events.append(payload)
            if prev is not None:
                pass
            n_tokens = _count_tokens_from_stream(events)
            if n_tokens == 0:
                results["errors"] += 1
                continue
            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000.0
            per_token_ms = total_ms / n_tokens if n_tokens else 0.0
            async with lock:
                results["ttft_ms"].append(ttft)
                results["per_token_ms"].append(per_token_ms)
                results["tokens"].append(n_tokens)
                results["requests"] += 1


async def run(url, concurrency, requests, max_tokens, password):
    results = {
        "pending": requests,
        "ttft_ms": [],
        "per_token_ms": [],
        "tokens": [],
        "requests": 0,
        "errors": 0,
    }
    lock = asyncio.Lock()
    t0 = time.perf_counter()
    await asyncio.gather(*[
        _one_client(url, _PROMPT, max_tokens, password, results, lock, i)
        for i in range(concurrency)
    ])
    wall = time.perf_counter() - t0
    total_tokens = sum(results["tokens"])
    n = results["requests"]
    ttft_ms = sorted(results["ttft_ms"]) if results["ttft_ms"] else [0.0]
    pt_ms = sorted(results["per_token_ms"]) if results["per_token_ms"] else [0.0]
    return {
        "concurrency": concurrency,
        "completed": n,
        "errors": results["errors"],
        "wall_s": round(wall, 2),
        "total_tokens": total_tokens,
        "tokens_per_s": round(total_tokens / wall, 1) if wall else 0.0,
        "p50_ttft_ms": round(ttft_ms[len(ttft_ms) // 2], 2) if ttft_ms else 0.0,
        "p50_per_token_ms": round(pt_ms[len(pt_ms) // 2], 2) if pt_ms else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=2,
                    help="total requests spread across the clients")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--password", default=None)
    ap.add_argument("--repeat", type=int, default=3, help="A/B repeats")
    args = ap.parse_args()

    env = asyncio.run(run(args.url, args.concurrency, args.requests,
                          args.max_tokens, args.password))
    print(json_dumps(env))


def json_dumps(d):
    import json
    return json.dumps(d, indent=2)


if __name__ == "__main__":
    main()