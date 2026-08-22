# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3 dense causal LM (Python model executor target).

Architecture: fused add + RMSNorm carrying (hidden, residual) between layers,
QK-norm before RoPE, gated-SiLU MLP. Tensor parallelism when tp_size>1.

Attention is delegated to the FlashInferBackend via the scoped ForwardContext.
The model does not import FlashInfer, own wrappers, or call plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    GatedMLP,
    HiddenParallelEmbedding,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
)
from xllm.python.model_executor.cp_utils import cp_merge_rows, cp_shard_positions, cp_shard_rows
from xllm.python.model_executor.forward_context import (
    get_forward_context,
    record_layer_event,
)  # noqa: F401
from xllm.python.models.base import PyModelBase


@dataclass
class Qwen3Config:
    hidden_size: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_position_embeddings: int = 40960
    vocab_size: int = 151936
    tie_word_embeddings: bool = True
    sliding_window: int = 0
    attention_bias: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3Config:
        def pick(*keys, default=None):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        hidden = int(pick("hidden_size", default=1024))
        n_heads = int(pick("n_heads", "num_attention_heads", default=16))
        return cls(
            hidden_size=hidden,
            n_layers=int(pick("n_layers", "num_hidden_layers", default=28)),
            n_heads=n_heads,
            n_kv_heads=int(pick("n_kv_heads", "num_key_value_heads", default=n_heads)),
            head_dim=int(pick("head_dim", default=hidden // n_heads)),
            intermediate_size=int(pick("intermediate_size", default=3072)),
            rms_norm_eps=float(pick("rms_norm_eps", default=1e-6)),
            rope_theta=float(pick("rope_theta", default=1e6)),
            max_position_embeddings=int(pick("max_position_embeddings", default=40960)),
            vocab_size=int(pick("vocab_size", default=151936)),
            tie_word_embeddings=bool(pick("tie_word_embeddings", default=True)),
            sliding_window=int(pick("sliding_window", default=0)),
            attention_bias=bool(pick("attention_bias", default=False)),
            tp_size=int(pick("tp_size", default=1)),
            tp_rank=int(pick("tp_rank", default=0)),
            dp_size=int(pick("dp_size", default=1)),
            dp_rank=int(pick("dp_rank", default=0)),
        )

    def _num_kv_head_replicas(self, tp_size: int) -> int:
        """Number of TP ranks that share (replicate) each KV head."""
        return max(1, tp_size // self.n_kv_heads)

    def head_split(self) -> tuple[int, int, int]:
        """Per-rank ``(num_heads, num_kv_heads, num_kv_head_replicas)``."""
        tp = self.tp_size
        assert self.n_heads % tp == 0, f"n_heads {self.n_heads} not divisible by tp_size {tp}"
        num_heads = self.n_heads // tp
        if self.n_kv_heads >= tp:
            assert self.n_kv_heads % tp == 0, f"n_kv_heads {self.n_kv_heads} not divisible by tp_size {tp}"
            num_kv_heads = self.n_kv_heads // tp
        else:
            assert tp % self.n_kv_heads == 0, f"tp_size {tp} not divisible by n_kv_heads {self.n_kv_heads}"
            num_kv_heads = 1
        return num_heads, num_kv_heads, self._num_kv_head_replicas(tp)

    def kv_shard(self, tp_rank: int, tp_size: int) -> tuple[int, int]:
        """Per-rank ``(kv_rank, kv_world)`` for sharding K/V projections."""
        replicas = self._num_kv_head_replicas(tp_size)
        return tp_rank // replicas, tp_size // replicas


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.layer_id = layer_id
        num_heads, num_kv_heads, replicas = cfg.head_split()
        tp = cfg.tp_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = cfg.head_dim
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim

        self.qkv_proj = ColumnParallelLinear(
            cfg.hidden_size,
            self.q_size + 2 * self.kv_size,
            tp,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            cfg.hidden_size,
            tp,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.head_dim**-0.5,
            sliding_window=cfg.sliding_window,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        mrope_section: list[int] | None = None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden)

        if mrope_section is not None and positions.dim() == 2:
            # mRoPE prefill: per-head Q/K RMSNorm (same math as the fused
            # kernel) then kernels.mrope, which does the time/height/width
            # section combination + rotation in one op.
            # cos_sin_cache here is the [max_pos, head_dim]=[cos_half|sin_half]
            # table; q/k stay 2D [N, num_heads*head_dim] as npu_mrope requires.
            num_tokens = qkv.size(0)
            q = torch.ops.xllm_ops.rms_norm(
                qkv[:, : self.q_size].reshape(num_tokens * self.num_heads, self.head_dim),
                self.q_norm.weight,
                self.q_norm.eps,
            ).view(num_tokens, self.q_size)
            k = torch.ops.xllm_ops.rms_norm(
                qkv[:, self.q_size : self.q_size + self.kv_size].reshape(num_tokens * self.num_kv_heads, self.head_dim),
                self.k_norm.weight,
                self.k_norm.eps,
            ).view(num_tokens, self.kv_size)
            v = qkv[:, self.q_size + self.kv_size :]
            q, k = kernels.mrope(
                positions,
                q,
                k,
                cos_sin_cache,
                self.head_dim,
                mrope_section=list(mrope_section),
                rotary_mode="half",
                cache_mode="interleave",
            )
        else:
            q, k, v = kernels.fused_qk_norm_rope(
                qkv,
                num_heads_q=self.num_heads,
                num_heads_k=self.num_kv_heads,
                num_heads_v=self.num_kv_heads,
                head_dim=self.head_dim,
                eps=self.q_norm.eps,
                q_weight=self.q_norm.weight,
                k_weight=self.k_norm.weight,
                cos_sin_cache=cos_sin_cache,
                position_ids=positions,
                cos=cos,
                sin=sin,
            )

        attn_out = self.attn(q, k, v)
        return self.o_proj(attn_out)


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Qwen3Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = Qwen3Attention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.mlp = GatedMLP(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.tp_size,
            dtype,
            device,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        mrope_section: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)

        hidden = self.self_attn(positions, hidden, cos_sin_cache, cos, sin, mrope_section)

        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        tp = cfg.tp_size
        assert cfg.hidden_size % tp == 0
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size // tp, tp, dtype=dtype, device=device
        )
        self.rotary = RotaryEmbedding(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList([Qwen3DecoderLayer(cfg, i, dtype, device) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mrope_section: list[int] | None = None,
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        # The fused QK-norm+RoPE kernel requires int64 position ids, but C++
        # passes them as int32. Cast once here instead of once per layer. In the
        # captured decode graph this single cast is recorded inside the graph
        # (its output lives in the graph memory pool), so replay re-casts the
        # updated static_positions correctly.
        positions = positions.to(torch.int64).contiguous()
        # Context-Parallel: shard the sequence across the CP group after embed
        # and merge back before the final norm (model-side CP semantics). Only
        # active on prefill with cp_size>1; cp_context is None otherwise.
        cp_context = get_forward_context().cp_context
        if cp_context is not None:
            hidden = cp_shard_rows(hidden, cp_context)
            positions = cp_shard_positions(positions, cp_context).contiguous()
        residual: torch.Tensor | None = None
        for i, layer in enumerate(self.layers):
            hidden, residual = layer(
                hidden,
                residual,
                positions,
                self.rotary.cos_sin_cache,
                None,
                None,
                mrope_section,
            )
            record_layer_event(i)
        hidden, _ = self.norm(hidden, residual)
        if cp_context is not None:
            hidden = cp_merge_rows(hidden, cp_context)
        return hidden


def load_qwen3_backbone_layer(
    layer_index: int,
    *,
    src_prefix: str,
    load_tensor: Callable[[str], torch.Tensor],
    shard: Callable[..., torch.Tensor],
    copy_in: Callable[[str, torch.Tensor], None],
    attention_bias: bool = False,
) -> None:
    """Load one Qwen3-dense decoder layer (shared by Qwen3, Qwen3-VL, and the
    DSpark draft). Reads ``src_prefix{i}.`` checkpoint keys, fuses q/k/v into
    qkv_proj and gate/up into gate_up_proj, and copies into ``model.layers.{i}.``
    params via the caller's closures. Callers own embed/norm/lm_head and any
    process_weights_after_loading.
    """
    src = f"{src_prefix}{layer_index}."
    dst = f"model.layers.{layer_index}."

    copy_in(dst + "input_layernorm.weight", load_tensor(src + "input_layernorm.weight"))
    copy_in(dst + "post_attention_layernorm.weight", load_tensor(src + "post_attention_layernorm.weight"))
    copy_in(dst + "self_attn.q_norm.weight", load_tensor(src + "self_attn.q_norm.weight"))
    copy_in(dst + "self_attn.k_norm.weight", load_tensor(src + "self_attn.k_norm.weight"))

    q = shard(src + "self_attn.q_proj.weight", dim=0)
    k = shard(src + "self_attn.k_proj.weight", dim=0, kv=True)
    v = shard(src + "self_attn.v_proj.weight", dim=0, kv=True)
    copy_in(dst + "self_attn.qkv_proj.weight", torch.cat([q, k, v], dim=0))
    copy_in(dst + "self_attn.o_proj.weight", shard(src + "self_attn.o_proj.weight", dim=1))

    if attention_bias:
        qb = shard(src + "self_attn.q_proj.bias", dim=0)
        kb = shard(src + "self_attn.k_proj.bias", dim=0, kv=True)
        vb = shard(src + "self_attn.v_proj.bias", dim=0, kv=True)
        copy_in(dst + "self_attn.qkv_proj.bias", torch.cat([qb, kb, vb], dim=0))
        # o_proj bias is replicated and added after the all-reduce, so every
        # rank loads the full (unsharded) bias.
        copy_in(dst + "self_attn.o_proj.bias", load_tensor(src + "self_attn.o_proj.bias"))

    gate = shard(src + "mlp.gate_proj.weight", dim=0)
    up = shard(src + "mlp.up_proj.weight", dim=0)
    copy_in(dst + "mlp.gate_up_proj.weight", torch.cat([gate, up], dim=0))
    copy_in(dst + "mlp.down_proj.weight", shard(src + "mlp.down_proj.weight", dim=1))


class Qwen3ForCausalLM(PyModelBase):
    """Top-level entry the C++ PyCausalLM drives."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = Qwen3Config.from_dict(config)
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.dtype = dtype
        self.device = device

        tp = self.cfg.tp_size
        dp = self.cfg.dp_size
        if tp * dp != int(config.get("world_size", tp * dp)):
            raise ValueError("world_size must equal tp_size * dp_size")
        if not 0 <= self.cfg.dp_rank < dp:
            raise ValueError("dp_rank must be in [0, dp_size)")
        assert self.cfg.vocab_size % tp == 0
        self.model = Qwen3Model(self.cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    # -- weight loading ---------------------------------------------------
    def load_weights(
        self,
        state_dicts: list,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        cfg = self.cfg

        kv_rank, kv_world = cfg.kv_shard(tp_rank, tp_size)

        def find(name: str):
            for sd in state_dicts:
                if sd.has(name):
                    return sd
            return None

        def load_tensor(name: str) -> torch.Tensor:
            sd = find(name)
            assert sd is not None, f"checkpoint tensor not found: {name}"
            return sd.get_tensor(name)

        def shard(name: str, dim: int, kv: bool = False) -> torch.Tensor:
            sd = find(name)
            assert sd is not None, f"checkpoint tensor not found: {name}"
            r = kv_rank if kv else tp_rank
            w = kv_world if kv else tp_size
            t = sd.get_tensor(name)
            if w <= 1:
                return t
            chunk_size = t.size(dim) // w
            return t.narrow(dim, r * chunk_size, chunk_size).contiguous()

        def copy_in(param_name: str, tensor: torch.Tensor) -> None:
            param = self.get_parameter(param_name)
            param.data.copy_(tensor.to(dtype=param.dtype, device=param.device))

        embed_name = "model.embed_tokens.weight"
        if not find(embed_name):
            embed_name = "embed_tokens.weight"
        copy_in("model.embed_tokens.weight", shard(embed_name, dim=1))

        for i in range(cfg.n_layers):
            load_qwen3_backbone_layer(
                i,
                src_prefix="model.layers.",
                load_tensor=load_tensor,
                shard=shard,
                copy_in=copy_in,
                attention_bias=cfg.attention_bias,
            )

            layer = self.model.layers[i]
            layer.self_attn.o_proj.process_weights_after_loading()
            layer.mlp.down_proj.process_weights_after_loading()

        norm_name = "model.norm.weight"
        if not find(norm_name):
            norm_name = "norm.weight"
        copy_in("model.norm.weight", load_tensor(norm_name))

        if cfg.tie_word_embeddings:
            lm_name = embed_name
        else:
            lm_name = "lm_head.weight"
        copy_in("lm_head.weight", shard(lm_name, dim=0))
