"""RWKV-7 inference entry point for pre-exported W8A16 checkpoints."""

import torch

from .quant.rwkv7_quant import QuantizedRWKV7
from .quant.w8a16 import gemm_w8a16


class RWKV_x070(QuantizedRWKV7):
    gemm = staticmethod(gemm_w8a16)
    quant_dtype = torch.int8
    quant_name = "W8A16"
