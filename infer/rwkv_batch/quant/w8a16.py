"""W8A16 quantization and CUDA GEMM interface.

Weights use symmetric, per-output-channel quantization.  The quantized weight
has the same ``[out_features, in_features]`` layout accepted by
``torch.nn.functional.linear``.
"""

import os

import torch
from torch.nn import functional as F


_extension_loaded = False


def _load_cuda_extension() -> None:
    global _extension_loaded
    if _extension_loaded:
        return
    from torch.utils.cpp_extension import load

    cuda_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cuda")
    load(
        name="rwkv_w8a16_cuda",
        sources=[
            os.path.join(cuda_dir, "w8a16_gemm.cpp"),
            os.path.join(cuda_dir, "w8a16_gemm.cu"),
        ],
        is_python_module=False,
        verbose=False,
        extra_include_paths=[
            os.path.join(cuda_dir, "third_party", "cutlass", "include")
        ],
        extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    )
    _extension_loaded = True


@torch.no_grad()
def quantize_w8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D floating-point weight to int8 and FP16 scales."""
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a 2-D floating-point tensor")

    max_abs = weight.float().abs().amax(dim=1)
    scale = (max_abs / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
    qweight = torch.round(weight.float() / scale[:, None]).clamp_(-127, 127)
    return qweight.to(torch.int8), scale.to(torch.float16)


def dequantize_w8(qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return an FP16 weight from an int8 weight and per-output scale."""
    if qweight.ndim != 2 or qweight.dtype != torch.int8:
        raise ValueError("qweight must be a 2-D int8 tensor")
    if scale.ndim != 1 or scale.numel() != qweight.shape[0]:
        raise ValueError("scale must contain one value per output feature")
    if scale.device != qweight.device:
        raise ValueError("qweight and scale must be on the same device")
    return qweight.to(torch.float16) * scale.to(torch.float16)[:, None]


def gemm_w8a16(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a W8A16 linear projection to ``x``.

    ``x`` may be a vector, ``[bsz, in_features]``, or have any number of
    leading dimensions. Activations, dequantized weights, and output are FP16.
    """
    if x.dtype != torch.float16:
        raise ValueError("W8A16 requires FP16 activations")
    if x.device != qweight.device:
        raise ValueError("x and qweight must be on the same device")
    if x.shape[-1] != qweight.shape[1]:
        raise ValueError("x and qweight have incompatible inner dimensions")
    if x.is_cuda:
        if bias is not None:
            bias = bias.to(device=x.device, dtype=torch.float16)
        _load_cuda_extension()
        return torch.ops.rwkv_w8a16.gemm(x, qweight, scale, bias)

    # Portable CPU fallback.
    weight = dequantize_w8(qweight, scale)
    if bias is not None:
        bias = bias.to(device=x.device, dtype=torch.float16)
    return F.linear(x, weight, bias)
