"""Export an RWKV-7 checkpoint with W8A16 weights.

Example:
    python -m infer.rwkv_batch.quant.export_quant \
        model.pth model-w8a16.pth --bits 8
"""

import argparse
from pathlib import Path

import torch

from .w8a16 import quantize_w8


# Only these large matrices are worth quantizing. The small low-rank matrices
# (w1/w2/a1/a2/v1/v2/g1/g2) stay floating point for both accuracy and speed.
_ATT_LINEAR_NAMES = {
    "receptance.weight", "key.weight", "value.weight", "output.weight",
}
_FFN_LINEAR_NAMES = {"key.weight", "value.weight"}
_TRANSPOSE_NAMES = {"w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2"}


def is_linear_weight(name: str) -> bool:
    if name == "head.weight":
        return True
    parts = name.split(".")
    if len(parts) < 4 or parts[0] != "blocks":
        return False
    leaf = ".".join(parts[3:])
    if parts[2] == "att":
        return leaf in _ATT_LINEAR_NAMES
    if parts[2] == "ffn":
        return leaf in _FFN_LINEAR_NAMES
    return False


def to_linear_layout(name: str, weight: torch.Tensor) -> torch.Tensor:
    """Convert RWKV low-rank matrices to PyTorch's [out, in] layout."""
    return weight.t() if name.rsplit(".", 1)[-1] in _TRANSPOSE_NAMES else weight


def needs_runtime_transpose(name: str) -> bool:
    """Return whether an unquantized RWKV low-rank matrix needs transposing."""
    return name.rsplit(".", 1)[-1] in _TRANSPOSE_NAMES


@torch.no_grad()
def export_quantized_checkpoint(input_path: str, output_path: str, bits: int = 8) -> None:
    if bits != 8:
        raise ValueError("only W8A16 is supported")
    state = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a state dict")

    output: dict[str, torch.Tensor] = {}
    count = 0
    original_bytes = 0
    quantized_bytes = 0
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint entry {name!r} is not a tensor")
        if is_linear_weight(name):
            weight = to_linear_layout(name, value.squeeze()).contiguous()
            qweight, scale = quantize_w8(weight)
            output[name] = qweight.cpu()
            output[name + ".scale"] = scale.cpu()
            count += 1
            original_bytes += value.numel() * value.element_size()
            quantized_bytes += qweight.numel() * qweight.element_size()
            quantized_bytes += scale.numel() * scale.element_size()
        else:
            # A number of RWKV checkpoints contain tensor views backed by much
            # larger storages. Saving the view directly serializes that entire
            # backing storage and can make the output several times too large.
            output[name] = value.clone()

    if count == 0:
        raise ValueError("no RWKV-7 linear weights were found")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    ratio = original_bytes / quantized_bytes
    print(f"exported {count} W{bits} matrices to {output_path}")
    print(f"quantized matrix storage: {quantized_bytes / 2**20:.2f} MiB ({ratio:.2f}x smaller)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="source FP checkpoint (.pth)")
    parser.add_argument("output", help="destination quantized checkpoint (.pth)")
    parser.add_argument("--bits", type=int, choices=(8,), default=8)
    args = parser.parse_args()
    export_quantized_checkpoint(args.input, args.output, args.bits)


if __name__ == "__main__":
    main()
