"""Export the large RWKV-7 projections in a packed GemLite format.

Example::

    python -m infer.rwkv_batch.quant.export_quant_gemlite \
        model.pth model-gemlite-int8.pth --format A16W8_INT8

The resulting checkpoint is consumed by ``rwkv7_quant_gemlite.py``. GemLite
packing is performed during conversion, so model startup does not quantize or
repack the weights.
"""

import argparse
from pathlib import Path

import torch

from .export_quant import is_linear_weight, to_linear_layout


FORMATS = ("A16W8_FP8", "A16W8_INT8", "A16W4_HQQ_INT")
FORMAT_CODES = {name: index for index, name in enumerate(FORMATS)}
FORMAT_KEY = "__gemlite_format__"
GROUP_SIZE_KEY = "__gemlite_group_size__"
VERSION_KEY = "__gemlite_checkpoint_version__"
STATE_INFIX = ".gemlite."
CHECKPOINT_VERSION = 2


def _require_gemlite():
    try:
        from gemlite.helper import A16W4_HQQ_INT, A16W8_FP8, A16W8_INT8
    except ImportError as exc:
        raise RuntimeError(
            "GemLite 0.6.x is required; install it with `pip install gemlite==0.6.0`"
        ) from exc
    return A16W8_FP8, A16W8_INT8, A16W4_HQQ_INT


@torch.inference_mode()
def quantize_hqq_int4(
    weight: torch.Tensor,
    group_size: int = 64,
    device: str = "cuda:0",
    optimize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unpacked HQQ-compatible uint4 values, scales, and zero points.

    Grouping follows HQQ ``axis=1``. The short proximal update is HQQ's
    calibration-free zero-point refinement; GemLite performs the final uint4
    packing after this function returns.
    """
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating-point matrix")
    if group_size < 16 or group_size % 8 != 0:
        raise ValueError("group_size must be a multiple of 8 and at least 16")
    if weight.shape[1] % group_size != 0:
        raise ValueError(
            f"in_features={weight.shape[1]} is not divisible by group_size={group_size}"
        )

    shape = weight.shape
    grouped = weight.to(device=device, dtype=torch.float32).reshape(-1, group_size)
    minimum = grouped.amin(dim=1, keepdim=True)
    maximum = grouped.amax(dim=1, keepdim=True)
    denominator = maximum - minimum
    inverse_scale = 15.0 / denominator
    inverse_scale = torch.where(
        denominator.abs() <= 1e-4,
        torch.ones_like(inverse_scale),
        inverse_scale,
    ).clamp(max=2e4)
    zero = torch.round(-minimum * inverse_scale)

    if optimize:
        # HQQ's legacy proximal solver minimizes an Lp reconstruction error by
        # refining the affine zero point. FP16 is intentional here and matches
        # HQQ's CUDA path while keeping conversion memory bounded.
        values = grouped.to(torch.float16)
        inverse_scale = inverse_scale.to(torch.float16)
        zero = zero.to(torch.float16)
        best_error = torch.tensor(torch.inf, device=values.device)
        beta = 10.0
        for _ in range(20):
            quantized = torch.round(values * inverse_scale + zero).clamp_(0, 15)
            reconstructed = (quantized - zero) / inverse_scale
            current_error = torch.abs(values - reconstructed).mean().float()
            if current_error >= best_error:
                break
            best_error = current_error
            error = values - reconstructed
            shrunk = torch.relu(
                torch.abs(error) - (1.0 / beta) * torch.abs(error).pow(-0.3)
            ) * torch.sign(error)
            zero = torch.mean(
                quantized - (values - shrunk) * inverse_scale,
                dim=1,
                keepdim=True,
            )
            beta *= 1.01

    # HQQ performs its final rounding against the original FP32 grouped
    # values, while using the FP16 scale/zero refined by the CUDA solver.
    quantized = torch.round(grouped * inverse_scale + zero).clamp_(0, 15)
    scales = inverse_scale.reciprocal()
    groups_per_row = shape[1] // group_size
    return (
        quantized.reshape(shape).to(torch.uint8),
        scales.reshape(shape[0], groups_per_row).to(torch.float16),
        zero.reshape(shape[0], groups_per_row).to(torch.float16),
    )


def _build_gemlite_linear(
    weight: torch.Tensor,
    quant_format: str,
    group_size: int,
    device: str,
    hqq_optimize: bool,
):
    A16W8_FP8, A16W8_INT8, A16W4_HQQ_INT = _require_gemlite()
    if quant_format == "A16W8_FP8":
        # W8 uses one scale per output channel. Applying that scale once after
        # accumulation avoids loading and multiplying an FP32 scale in every K
        # tile. This is mathematically equivalent to pre-scaling each weight.
        return A16W8_FP8(
            device=device, dtype=torch.float16, post_scale=True
        ).from_weights(weight)
    if quant_format == "A16W8_INT8":
        return A16W8_INT8(
            device=device, dtype=torch.float16, post_scale=True
        ).from_weights(weight)
    if quant_format == "A16W4_HQQ_INT":
        qweight, scales, zeros = quantize_hqq_int4(
            weight, group_size=group_size, device=device, optimize=hqq_optimize
        )
        return A16W4_HQQ_INT(device=device, dtype=torch.float16).from_weights(
            qweight, scales, zeros
        )
    raise ValueError(f"unsupported GemLite format: {quant_format}")


@torch.inference_mode()
def export_quantized_checkpoint(
    input_path: str,
    output_path: str,
    quant_format: str,
    group_size: int = 64,
    device: str = "cuda:0",
    hqq_optimize: bool = True,
) -> None:
    if quant_format not in FORMAT_CODES:
        raise ValueError(f"format must be one of {FORMATS}")
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("CUDA is required to export packed GemLite weights")

    state = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a state dict")

    output: dict[str, torch.Tensor] = {
        FORMAT_KEY: torch.tensor(FORMAT_CODES[quant_format], dtype=torch.int32),
        GROUP_SIZE_KEY: torch.tensor(
            group_size if quant_format == "A16W4_HQQ_INT" else 0,
            dtype=torch.int32,
        ),
        VERSION_KEY: torch.tensor(CHECKPOINT_VERSION, dtype=torch.int32),
    }
    count = 0
    quantized_bytes = 0
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint entry {name!r} is not a tensor")
        if not is_linear_weight(name):
            # Break shared storage views so torch.save does not serialize a
            # much larger backing tensor for a small RWKV parameter.
            output[name] = value.clone()
            continue

        weight = to_linear_layout(name, value.squeeze()).contiguous()
        layer = _build_gemlite_linear(
            weight,
            quant_format=quant_format,
            group_size=group_size,
            device=device,
            hqq_optimize=hqq_optimize,
        )
        for key, tensor in layer.state_dict().items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"GemLite state {name}.{key} is not a tensor")
            saved = tensor.detach().cpu().clone()
            output[name + STATE_INFIX + key] = saved
            quantized_bytes += saved.numel() * saved.element_size()
        count += 1
        del layer, weight
        torch.cuda.empty_cache()
        print(f"[{count}] packed {name}", flush=True)

    if count == 0:
        raise ValueError("no RWKV-7 large linear weights were found")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"exported {count} {quant_format} matrices to {output_path}")
    print(f"packed GemLite tensor storage: {quantized_bytes / 2**20:.2f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="source FP checkpoint (.pth)")
    parser.add_argument("output", help="destination packed GemLite checkpoint (.pth)")
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--no-hqq-optimize",
        action="store_true",
        help="disable HQQ proximal zero-point refinement for faster W4 export",
    )
    args = parser.parse_args()
    export_quantized_checkpoint(
        args.input,
        args.output,
        quant_format=args.format,
        group_size=args.group_size,
        device=args.device,
        hqq_optimize=not args.no_hqq_optimize,
    )


if __name__ == "__main__":
    main()
