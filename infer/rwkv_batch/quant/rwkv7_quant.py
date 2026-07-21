"""RWKV-7 implementation for pre-exported W8A16 checkpoints."""

from typing import Callable

import torch
from torch.nn import functional as F

from ..rwkv7 import (
    DTYPE,
    HEAD_SIZE,
    RWKV7_BATCH_OP,
    RWKV7_ONE_OP,
    RWKV7_SEQ_OP,
    RWKV_x070 as FP16RWKV,
)
from .export_quant import is_linear_weight, needs_runtime_transpose


Gemm = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


class QuantizedRWKV7(torch.nn.Module):
    """RWKV-7 inference using a pre-exported weight-only checkpoint."""

    gemm: Gemm
    quant_dtype: torch.dtype
    quant_name: str

    def __init__(self, args):
        super().__init__()
        self.args = args
        args.head_size = HEAD_SIZE
        self.eval()

        self.z = torch.load(args.MODEL_NAME + ".pth", map_location="cpu", weights_only=True)
        z = self.z
        if not isinstance(z, dict):
            raise ValueError("quantized checkpoint must contain a state dict")
        if "blocks.0.att.r_k" not in z:
            raise ValueError("checkpoint is not an RWKV-7 model")
        self.n_head, self.head_size = z["blocks.0.att.r_k"].shape
        args.n_embd = self.n_head * self.head_size
        if self.head_size != HEAD_SIZE:
            raise ValueError(f"expected head size {HEAD_SIZE}, got {self.head_size}")

        max_layer = -1
        for name in list(z):
            value = z[name]
            if name.endswith(".scale"):
                value = value.squeeze().to(dtype=DTYPE, device="cuda")
            elif is_linear_weight(name):
                scale_name = name + ".scale"
                if value.dtype != self.quant_dtype or scale_name not in z:
                    raise ValueError(
                        f"{name} is not a valid {self.quant_name} weight; "
                        "export the model with infer.rwkv_batch.quant.export_quant"
                    )
                value = value.to(device="cuda")
            else:
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
            args.vocab_size = z["head.weight"].shape[0]
        self.n_layer, self.n_embd = args.n_layer, args.n_embd
        self.prefill_chunk_size = 256

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

    # Reuse the batching/state management code; the three compute entry points
    # below dispatch all matrix multiplications through quantized GEMM.
    refresh_max_prefill_bsz = FP16RWKV.refresh_max_prefill_bsz
    generate_zero_state = FP16RWKV.generate_zero_state
    forward = FP16RWKV.forward
    forward_batch = FP16RWKV.forward_batch
    forward_batch_same_length = FP16RWKV.forward_batch_same_length

    def _linear(
        self,
        x: torch.Tensor,
        name: str,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if is_linear_weight(name):
            return self.gemm(x, self.z[name], self.z[name + ".scale"], bias)
        return F.linear(x, self.z[name], bias)

    def _tmix_one(self, layer_id, x, x_prev, v_first, state, elapsed_t):
        z = self.z
        att = f"blocks.{layer_id}.att."
        H, N = self.n_head, self.head_size
        xx = x_prev[0] - x
        x_prev[0] = x
        xr = x + xx * z[att + "x_r"]
        xw = x + xx * z[att + "x_w"]
        xk = x + xx * z[att + "x_k"]
        xv = x + xx * z[att + "x_v"]
        xa = x + xx * z[att + "x_a"]
        xg = x + xx * z[att + "x_g"]

        r = self._linear(xr, att + "receptance.weight")
        w = self._linear(torch.tanh(self._linear(xw, att + "w1")), att + "w2", z[att + "w0"])
        k = self._linear(xk, att + "key.weight")
        v = self._linear(xv, att + "value.weight")
        a = torch.sigmoid(self._linear(self._linear(xa, att + "a1"), att + "a2", z[att + "a0"]))
        g = self._linear(torch.sigmoid(self._linear(xg, att + "g1")), att + "g2")
        kk = F.normalize((k * z[att + "k_k"]).view(H, N), dim=-1, p=2.0).view(H * N)
        k = k * (1 + (a - 1) * z[att + "k_a"])
        if layer_id == 0:
            v_first = v
        else:
            gate = torch.sigmoid(self._linear(self._linear(xv, att + "v1"), att + "v2", z[att + "v0"]))
            v = v + (v_first - v) * gate
        out = RWKV7_ONE_OP(state, r, w, k, v, -kk, kk * a, elapsed_t)
        out = F.group_norm(out.view(1, H * N), H, z[att + "ln_x.weight"], z[att + "ln_x.bias"], 64e-5).view(H * N)
        out = out + ((r * k * z[att + "r_k"]).view(H, N).sum(-1, keepdim=True) * v.view(H, N)).view(H * N)
        return self._linear(out * g, att + "output.weight"), v_first

    def _tmix_seq(self, layer_id, x, x_prev, v_first, state, elapsed_t):
        z = self.z
        att = f"blocks.{layer_id}.att."
        T, H, N = x.shape[0], self.n_head, self.head_size
        xx = torch.cat((x_prev[0].unsqueeze(0), x[:-1])) - x
        x_prev[0] = x[-1]
        xr = x + xx * z[att + "x_r"]
        xw = x + xx * z[att + "x_w"]
        xk = x + xx * z[att + "x_k"]
        xv = x + xx * z[att + "x_v"]
        xa = x + xx * z[att + "x_a"]
        xg = x + xx * z[att + "x_g"]
        r = self._linear(xr, att + "receptance.weight")
        w = self._linear(torch.tanh(self._linear(xw, att + "w1")), att + "w2", z[att + "w0"])
        k = self._linear(xk, att + "key.weight")
        v = self._linear(xv, att + "value.weight")
        a = torch.sigmoid(self._linear(self._linear(xa, att + "a1"), att + "a2", z[att + "a0"]))
        g = self._linear(torch.sigmoid(self._linear(xg, att + "g1")), att + "g2")
        kk = F.normalize((k * z[att + "k_k"]).view(T, H, N), dim=-1, p=2.0).view(T, H * N)
        k = k * (1 + (a - 1) * z[att + "k_a"])
        if layer_id == 0:
            v_first = v
        else:
            gate = torch.sigmoid(self._linear(self._linear(xv, att + "v1"), att + "v2", z[att + "v0"]))
            v = v + (v_first - v) * gate
        out = RWKV7_SEQ_OP(state, r, w, k, v, -kk, kk * a, elapsed_t)
        out = F.group_norm(out.view(T, H * N), H, z[att + "ln_x.weight"], z[att + "ln_x.bias"], 64e-5).view(T, H * N)
        out = out + ((r * k * z[att + "r_k"]).view(T, H, N).sum(-1, keepdim=True) * v.view(T, H, N)).view(T, H * N)
        return self._linear(out * g, att + "output.weight"), v_first

    def _tmix_batch(self, layer_id, x, x_prev, v_first, state, elapsed_t):
        z = self.z
        att = f"blocks.{layer_id}.att."
        B, T, C = x.shape
        H, N = self.n_head, self.head_size
        xx = torch.cat((x_prev[0].unsqueeze(1), x[:, :-1]), dim=1) - x
        x_prev[0] = x[:, -1]
        xr = x + xx * z[att + "x_r"]
        xw = x + xx * z[att + "x_w"]
        xk = x + xx * z[att + "x_k"]
        xv = x + xx * z[att + "x_v"]
        xa = x + xx * z[att + "x_a"]
        xg = x + xx * z[att + "x_g"]
        r = self._linear(xr, att + "receptance.weight")
        w = self._linear(torch.tanh(self._linear(xw, att + "w1")), att + "w2", z[att + "w0"])
        k = self._linear(xk, att + "key.weight")
        v = self._linear(xv, att + "value.weight")
        a = torch.sigmoid(self._linear(self._linear(xa, att + "a1"), att + "a2", z[att + "a0"]))
        g = self._linear(torch.sigmoid(self._linear(xg, att + "g1")), att + "g2")
        kk = F.normalize((k * z[att + "k_k"]).view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * z[att + "k_a"])
        if layer_id == 0:
            v_first = v
        else:
            gate = torch.sigmoid(self._linear(self._linear(xv, att + "v1"), att + "v2", z[att + "v0"]))
            v = v + (v_first - v) * gate
        out = RWKV7_BATCH_OP(state, r, w, k, v, -kk, kk * a, elapsed_t).view(B * T, C)
        out = F.group_norm(out, H, z[att + "ln_x.weight"], z[att + "ln_x.bias"], 64e-5).view(B, T, C)
        out = out + ((r * k * z[att + "r_k"]).view(B, T, H, N).sum(-1, keepdim=True) * v.view(B, T, H, N)).view(B, T, C)
        return self._linear(out * g, att + "output.weight"), v_first

    def _cmix(self, layer_id, x, x_prev, mode):
        z = self.z
        ffn = f"blocks.{layer_id}.ffn."
        if mode == "one":
            xx = x_prev[1] - x
            x_prev[1] = x
        elif mode == "seq":
            xx = torch.cat((x_prev[1].unsqueeze(0), x[:-1])) - x
            x_prev[1] = x[-1]
        else:
            xx = torch.cat((x_prev[1].unsqueeze(1), x[:, :-1]), dim=1) - x
            x_prev[1] = x[:, -1]
        k = torch.relu(self._linear(x + xx * z[ffn + "x_k"], ffn + "key.weight")) ** 2
        return self._linear(k, ffn + "value.weight")

    @torch.no_grad()
    def forward_one(self, x, state):
        v_first = torch.empty_like(x)
        for i in range(self.n_layer):
            block = f"blocks.{i}."
            xx = F.layer_norm(x, (self.n_embd,), self.z[block + "ln1.weight"], self.z[block + "ln1.bias"])
            xx, v_first = self._tmix_one(i, xx, state[0][i], v_first, state[1][i], state[2])
            x = x + xx
            xx = F.layer_norm(x, (self.n_embd,), self.z[block + "ln2.weight"], self.z[block + "ln2.bias"])
            x = x + self._cmix(i, xx, state[0][i], "one")
        x = F.layer_norm(x, (self.n_embd,), self.z["ln_out.weight"], self.z["ln_out.bias"])
        state[2] += 1
        return self._linear(x, "head.weight")

    @torch.no_grad()
    def forward_seq(self, idx, state, full_output=False):
        x = self.z["emb.weight"][idx]
        v_first = torch.empty_like(x)
        for i in range(self.n_layer):
            block = f"blocks.{i}."
            xx = F.layer_norm(x, (self.n_embd,), self.z[block + "ln1.weight"], self.z[block + "ln1.bias"])
            xx, v_first = self._tmix_seq(i, xx, state[0][i], v_first, state[1][i], state[2])
            x = x + xx
            xx = F.layer_norm(x, (self.n_embd,), self.z[block + "ln2.weight"], self.z[block + "ln2.bias"])
            x = x + self._cmix(i, xx, state[0][i], "seq")
        if not full_output:
            x = x[-1]
        x = F.layer_norm(x, (self.n_embd,), self.z["ln_out.weight"], self.z["ln_out.bias"])
        state[2] += len(idx)
        return self._linear(x, "head.weight")

    @torch.no_grad()
    def forward_seq_batch(self, idxs, state, full_output=False):
        x = self.z["emb.weight"][torch.tensor(idxs, device="cuda")]
        v_first = torch.empty_like(x)
        for i in range(self.n_layer):
            block = f"blocks.{i}."
            xx = F.layer_norm(x, (self.n_embd,), self.z[block + "ln1.weight"], self.z[block + "ln1.bias"])
            xx, v_first = self._tmix_batch(i, xx, state[0][i], v_first, state[1][i], state[2])
            x = x + xx
            xx = F.layer_norm(x, (self.n_embd,), self.z[block + "ln2.weight"], self.z[block + "ln2.bias"])
            x = x + self._cmix(i, xx, state[0][i], "batch")
        if not full_output:
            x = x[:, -1]
        x = F.layer_norm(x, (self.n_embd,), self.z["ln_out.weight"], self.z["ln_out.bias"])
        state[2] += len(idxs[0])
        return self._linear(x, "head.weight")
