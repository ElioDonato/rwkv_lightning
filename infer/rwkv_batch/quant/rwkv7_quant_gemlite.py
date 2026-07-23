"""RWKV-7 inference backed by pre-packed GemLite linear projections."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ..rwkv7 import DTYPE, HEAD_SIZE
from .export_quant import needs_runtime_transpose
from .export_quant_gemlite import (
    FORMATS,
    FORMAT_KEY,
    GROUP_SIZE_KEY,
    CHECKPOINT_VERSION,
    STATE_INFIX,
    VERSION_KEY,
)
from .rwkv7_quant import QuantizedRWKV7


def _require_gemlite():
    try:
        import gemlite
        from gemlite import GemLiteLinear
        from gemlite.core import GEMLITE_TRITON_MAPPING, get_matmul_type
        from gemlite.dtypes import DType, is_mx_dtype
    except ImportError as exc:
        raise RuntimeError(
            "GemLite 0.6.x is required; install it with `pip install gemlite==0.6.0`"
        ) from exc
    version = getattr(gemlite, "__version__", "unknown")
    if not version.startswith("0.6."):
        raise RuntimeError(
            f"GemLite 0.6.x is required for this packed checkpoint, got {version}"
        )
    return GemLiteLinear, GEMLITE_TRITON_MAPPING, get_matmul_type, DType, is_mx_dtype


@dataclass(frozen=True)
class _GemLiteRuntime:
    """Cached arguments for one packed projection.

    GemLite 0.6.0's public functional entry point reads data_contiguous from
    meta_args[1] (W_nbits) instead of the final metadata field. Calling the
    same GemLite Triton kernels here with the correct flag both fixes W8's
    column-major layout hint and avoids rebuilding argument lists 193 times
    per decoded token.
    """

    tensor_args: tuple[torch.Tensor, ...]
    meta_args: tuple[int, ...]
    bias: torch.Tensor | None
    out_features: int


class RWKV_x070(QuantizedRWKV7):
    """Load a checkpoint produced by ``export_quant_gemlite.py``."""

    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args
        args.head_size = HEAD_SIZE
        self.eval()

        z = torch.load(args.MODEL_NAME + ".pth", map_location="cpu", weights_only=True)
        if not isinstance(z, dict):
            raise ValueError("quantized checkpoint must contain a state dict")
        if FORMAT_KEY not in z or VERSION_KEY not in z:
            raise ValueError(
                "not a packed GemLite checkpoint; use "
                "infer.rwkv_batch.quant.export_quant_gemlite"
            )
        version = int(z.pop(VERSION_KEY).item())
        if version != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported GemLite checkpoint version: {version}")
        format_code = int(z.pop(FORMAT_KEY).item())
        if not 0 <= format_code < len(FORMATS):
            raise ValueError(f"invalid GemLite format code: {format_code}")
        self.quant_name = "GemLite " + FORMATS[format_code]
        self.group_size = int(z.pop(GROUP_SIZE_KEY).item())

        (
            GemLiteLinear,
            self._gemlite_kernels,
            self._gemlite_matmul_type,
            self._gemlite_dtype,
            self._gemlite_is_mx_dtype,
        ) = _require_gemlite()
        suffix = STATE_INFIX + "W_q"
        quant_names = sorted(key[: -len(suffix)] for key in z if key.endswith(suffix))
        if not quant_names:
            raise ValueError("checkpoint does not contain packed GemLite weights")

        # Register packed layers as real submodules. Dots are not valid
        # ModuleDict keys, so keep a compact name-to-ModuleList index mapping.
        self.gemlite_linears = torch.nn.ModuleList()
        self._gemlite_indices: dict[str, int] = {}
        self._gemlite_runtime: list[_GemLiteRuntime] = []
        for name in quant_names:
            prefix = name + STATE_INFIX
            packed_state = {
                key[len(prefix):]: z.pop(key).to(device="cuda")
                for key in list(z)
                if key.startswith(prefix)
            }
            layer = GemLiteLinear()
            layer.load_state_dict(packed_state)
            layer.eval()
            index = len(self.gemlite_linears)
            self._gemlite_indices[name] = index
            self.gemlite_linears.append(layer)
            self._gemlite_runtime.append(
                _GemLiteRuntime(
                    tensor_args=tuple(layer.get_tensor_args()),
                    meta_args=tuple(layer.get_meta_args()),
                    bias=layer.bias,
                    out_features=layer.out_features,
                )
            )
            # Preserve familiar checkpoint keys without duplicating storage.
            z[name] = layer.W_q
            z[name + ".scale"] = layer.scales
            if FORMATS[format_code] == "A16W4_HQQ_INT":
                z[name + ".zero"] = layer.zeros

        if "blocks.0.att.r_k" not in z:
            raise ValueError("checkpoint is not an RWKV-7 model")
        self.n_head, self.head_size = z["blocks.0.att.r_k"].shape
        args.n_embd = self.n_head * self.head_size
        if self.head_size != HEAD_SIZE:
            raise ValueError(f"expected head size {HEAD_SIZE}, got {self.head_size}")

        max_layer = -1
        packed_names = set(quant_names)
        for name in list(z):
            if name in packed_names or name.endswith((".scale", ".zero")):
                continue
            value = z[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"checkpoint entry {name!r} is not a tensor")
            if needs_runtime_transpose(name):
                value = value.t()
            value = value.squeeze().to(dtype=DTYPE, device="cuda")
            if name.endswith("att.r_k"):
                value = value.flatten()
            z[name] = value.contiguous()
            parts = name.split(".")
            if parts[0] == "blocks":
                max_layer = max(max_layer, int(parts[1]))

        args.n_layer = max_layer + 1
        if not hasattr(args, "vocab_size"):
            args.vocab_size = self._gemlite_runtime[
                self._gemlite_indices["head.weight"]
            ].out_features
        self.n_layer, self.n_embd = args.n_layer, args.n_embd
        self.prefill_chunk_size = 256
        self.z = z

        z["emb.weight"] = F.layer_norm(
            z["emb.weight"],
            (args.n_embd,),
            weight=z["blocks.0.ln0.weight"],
            bias=z["blocks.0.ln0.bias"],
        )
        z["blocks.0.att.v0"] = z["blocks.0.att.a0"]
        z["blocks.0.att.v1"] = z["blocks.0.att.a1"]
        z["blocks.0.att.v2"] = z["blocks.0.att.a2"]

        self.refresh_max_prefill_bsz()
        self.max_prefill_bsz_limit = int(self.max_prefill_bsz)
        print(args)
        print(
            f"loaded {self.quant_name}; max_prefill_bsz={self.max_prefill_bsz} "
            f"for prefill_chunk_size={self.prefill_chunk_size}"
        )

    def _linear(
        self,
        x: torch.Tensor,
        name: str,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        index = self._gemlite_indices.get(name)
        if index is not None:
            output = self._gemlite_forward(x, self._gemlite_runtime[index])
            return output if bias is None else output + bias
        return F.linear(x, self.z[name], bias)

    def _gemlite_forward(
        self,
        x: torch.Tensor,
        runtime: _GemLiteRuntime,
    ) -> torch.Tensor:
        """Call GemLite 0.6 kernels with its data-contiguity bug corrected."""
        meta = runtime.meta_args
        if meta[0]:
            # The three supported A16 formats never dynamically quantize x.
            # Keep this guard so a future activation-quantized checkpoint does
            # not silently take an invalid fast path.
            raise RuntimeError("scaled GemLite activations are not supported")
        if not x.is_contiguous():
            x = x.contiguous()

        original_shape = x.shape
        x = x.view(-1, x.shape[-1])
        input_dtype = meta[5]
        if input_dtype == self._gemlite_dtype.BF16.value:
            input_dtype = self._gemlite_dtype.FP16.value
        elif input_dtype == self._gemlite_dtype.MXBF16.value:
            input_dtype = self._gemlite_dtype.MXFP16.value
        type_id = input_dtype * 100 + meta[1]
        matmul_type = self._gemlite_matmul_type(
            x.shape[0], meta[1], self._gemlite_is_mx_dtype(input_dtype)
        )
        meta_scale: float | torch.Tensor = 0.0
        if meta[2] == 16:
            # Match GemLite's weight-only NVFP4 special case. The currently
            # supported INT/FP8 formats normally leave this at zero.
            meta_scale = runtime.tensor_args[3]
        output = self._gemlite_kernels[matmul_type].forward(
            x,
            *runtime.tensor_args[:3],
            None,
            *meta[1:-1],
            bool(meta[-1]),
            type_id,
            meta_scale=meta_scale,
        )
        output = output.view(original_shape[:-1] + (runtime.out_features,))
        if runtime.bias is not None:
            output = output + runtime.bias
        return output
