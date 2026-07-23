"""Convenience entry point for end-to-end GemLite RWKV-7 experiments.

All packed formats supported by export_quant_gemlite.py are auto-detected.
In particular, A16W8_INT8 stays on GemLite and is never redirected to the
native CUDA W8A16 backend.
"""

from .quant.rwkv7_quant_gemlite import RWKV_x070

__all__ = ["RWKV_x070"]
