import logging

logger = logging.getLogger("model.loader")

import re
import types
import torch

from infer.rwkv_batch.utils import TRIE_TOKENIZER


INFERENCE_ENGINES = ("fp16", "gemlite", "cutlass")


def _get_model_class(inference_engine: str):
    if inference_engine == "fp16":
        from infer.rwkv_batch.rwkv7 import RWKV_x070
    elif inference_engine == "gemlite":
        from infer.rwkv_batch.rwkv7_w8a16_gemlite import RWKV_x070
    elif inference_engine == "cutlass":
        from infer.rwkv_batch.rwkv7_w8a16 import RWKV_x070
    else:
        raise ValueError(
            f"unsupported inference engine {inference_engine!r}; "
            f"expected one of {INFERENCE_ENGINES}"
        )
    return RWKV_x070


def load_model_and_tokenizer(model_path: str, inference_engine: str = "fp16"):
    rocm_flag = torch.version.hip is not None
    if inference_engine not in INFERENCE_ENGINES:
        raise ValueError(
            f"unsupported inference engine {inference_engine!r}; "
            f"expected one of {INFERENCE_ENGINES}"
        )
    if rocm_flag and inference_engine == "cutlass":
        raise RuntimeError("the CUTLASS inference engine requires NVIDIA CUDA")

    logger.info(f"\n[INFO] Loading RWKV-7 model from {model_path} "
        f"with inference engine {inference_engine}\n")

    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.head_size = 64
    args.inference_engine = inference_engine
    if model_path.endswith(".pth"):
        args.MODEL_NAME = re.sub(r"\.pth$", "", model_path)
    else:
        args.MODEL_NAME = model_path

    model_class = _get_model_class(inference_engine)
    model = model_class(args)
    tokenizer = TRIE_TOKENIZER("infer/rwkv_batch/rwkv_vocab_v20230424.txt")

    logger.info("[INFO] Model loaded successfully.\n")

    return model, tokenizer, args, rocm_flag
