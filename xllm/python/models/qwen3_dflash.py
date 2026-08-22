# Copyright 2026 The xLLM Authors.
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

"""DFlash draft-model base for the Python executor.

Extends the Qwen3 backbone with the DFlash-family surface:

  * ``fc`` + ``hidden_norm``: project concatenated target aux hidden states
    ([tokens, hidden * num_captured]) into the draft's KV-context space. The
    real in-dim is checkpoint-authoritative and reset at weight load.
  * ``write_context_kv``: attention-free per-layer paged K/V scatter driven by
    the projected aux hidden states.
  * Non-causal attention: every draft position attends the whole context block
    (backend picks sparse_mode=NONE, atten_mask=None on the prefill path).

DSpark inherits this and adds Markov / confidence heads on ``ForCausalLM``.
Mirrors the native ``DFlashQwen3ModelImpl : QWen3ModelImpl`` inheritance
(``xllm/models/llm/npu/qwen3_dflash.h:36``).
"""

from __future__ import annotations

from typing import Callable

import torch

from xllm.python import kernels
from xllm.python.layers import ColumnParallelLinear, RMSNorm
from xllm.python.model_executor.forward_context import LayerSynchronizer
from xllm.python.models.qwen3 import Qwen3Config, Qwen3Model, load_qwen3_backbone_layer


class DFlashModel(Qwen3Model):
    """DFlash draft backbone: Qwen3 stack + aux-hidden projection."""

    def __init__(self, cfg: Qwen3Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(cfg, dtype, device)
        # fc's real in-dim is checkpoint-authoritative (load_weights rebuilds
        # from fc.weight.shape); this placeholder is replaced during load.
        self.fc = ColumnParallelLinear(cfg.hidden_size, cfg.hidden_size, 1, bias=False, dtype=dtype, device=device)
        self.hidden_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        # DFlash/DSpark draft attends the whole context block non-causally.
        for layer in self.layers:
            layer.self_attn.attn.causal = False

    # forward inherited from Qwen3Model (mrope=None default; causal=False set on
    # attention layers drives sparse_mode selection in the backend).

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        layer_caches: list,
        layer_synchronizer: LayerSynchronizer | None = None,
    ) -> bool:
        """Project target aux hidden into every draft layer's paged K/V cache.

        Attention-free, runs eager (outside the decode graph). Only K/V are
        projected; the fused kernel's zeroed-Q slot is safe because each head
        is normed independently. Returns False on layer_synchronizer failure so
        the caller can fail-fast the PD-PUSH scatter.

        Performance choice: DSpark draft is typically ~5 layers, where per-layer
        ``fused_qk_norm_rope`` (single NPU kernel doing rms_norm + rope on the
        [Q|K|V] scratch) beats a cross-layer fused K/V GEMM + hand-rolled
        RMSNorm + RoPE (which loses the fused kernel and adds fp32
        variance/rsqrt + cos/sin concat ops).
        """
        projected = self.hidden_norm(self.fc(target_hidden))
        positions = positions.to(torch.int64).contiguous()
        cos_sin_cache = self.rotary.cos_sin_cache
        # Reused [Q|K|V] scratch: Q stays zero, only K/V is rewritten per layer.
        # All layers share qkv shape by construction (built from the same cfg).
        num_heads = self._num_heads
        num_kv_heads = self._num_kv_heads
        head_dim = self._head_dim
        eps = self._k_norm_eps
        q_size = self._q_size
        qkv = projected.new_empty(projected.shape[0], self._qkv_dim)
        qkv[:, :q_size].zero_()
        kv_view = qkv[:, q_size:]
        record_event = None if layer_synchronizer is None else layer_synchronizer.record_event
        kv_shape = (projected.shape[0], num_kv_heads, head_dim)
        for i in range(len(self.layers)):
            q_weight, k_weight = self._norm_weights[i]
            torch.matmul(projected, self._kv_weights_t[i], out=kv_view)
            _, k, v = kernels.fused_qk_norm_rope(
                qkv,
                num_heads_q=num_heads,
                num_heads_k=num_kv_heads,
                num_heads_v=num_kv_heads,
                head_dim=head_dim,
                eps=eps,
                q_weight=q_weight,
                k_weight=k_weight,
                cos_sin_cache=cos_sin_cache,
                position_ids=positions,
                cos=None,
                sin=None,
            )
            k_cache, v_cache = layer_caches[i]
            kernels.reshape_paged_cache(cache_slots, k.view(kv_shape), v.view(kv_shape), k_cache, v_cache)
            if record_event is not None and not record_event(i):
                return False
        return True

    def load_backbone_layers(
        self,
        *,
        src_prefix: str,
        load_tensor: Callable[[str], torch.Tensor],
        shard: Callable[..., torch.Tensor],
        copy_in: Callable[[str, torch.Tensor], None],
    ) -> None:
        """Load the Qwen3-dense stack from checkpoint (draft prefix ``layers.``,
        full model prefix ``model.layers.``). Runs each layer's
        ``process_weights_after_loading`` after weight copy.
        """
        # write_context_kv hot loop reuses these transposed KV-projection views.
        self._kv_weights_t: list[torch.Tensor] = []
        # Per-layer (q_norm.weight, k_norm.weight) captured once; the write
        # loop does not retraverse nn.Module attribute chains per step.
        self._norm_weights: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(self.cfg.n_layers):
            load_qwen3_backbone_layer(
                i,
                src_prefix=src_prefix,
                load_tensor=load_tensor,
                shard=shard,
                copy_in=copy_in,
            )
            attn = self.layers[i].self_attn
            attn.o_proj.process_weights_after_loading()
            self.layers[i].mlp.down_proj.process_weights_after_loading()
            self._kv_weights_t.append(attn.qkv_proj.weight[attn.num_heads * attn.head_dim :].t())
            self._norm_weights.append((attn.q_norm.weight, attn.k_norm.weight))
        # Per-layer [Q|K|V] shape and RMSNorm eps are shared across layers; cache
        # scalars so write_context_kv skips per-step attribute walks.
        attn0 = self.layers[0].self_attn
        self._num_heads = attn0.num_heads
        self._num_kv_heads = attn0.num_kv_heads
        self._head_dim = attn0.head_dim
        self._k_norm_eps = attn0.k_norm.eps
        self._q_size = attn0.num_heads * attn0.head_dim
        self._qkv_dim = attn0.qkv_proj.weight.shape[0]
