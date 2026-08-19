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

"""DSpark draft model for the Python executor.

Extends the DFlash backbone (see ``dflash``) with Markov + optional
Confidence heads. Mirrors the native ``DSparkQwen3ForCausalLMImpl`` layout
(heads on the ForCausalLM wrapper, backbone reused from DFlash).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python.layers import ColumnParallelLinear
from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v32 import W8A8WeightLoader
from xllm.python.models.qwen3 import Qwen3Config
from xllm.python.models.qwen3_dflash import DFlashModel


class DSparkMarkovHead(nn.Module):
    """Low-rank Markov logit-bias:  bias(prev) = w2 @ w1[prev]."""

    def __init__(
        self,
        vocab_size: int,
        markov_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank, dtype=dtype, device=device)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False, dtype=dtype, device=device)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.embed(token_ids))


class DSparkConfidenceHead(nn.Module):
    """Per-position acceptance-probability head (fp32 projection).

    Accept-probability precision drives the adaptive-speculative budget, so
    the projection runs in fp32 (matches native dspark_confidence_head).
    """

    def __init__(
        self,
        hidden_size: int,
        markov_rank: int,
        with_markov: bool,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.with_markov = with_markov
        input_size = hidden_size + markov_rank if with_markov else hidden_size
        self.proj = nn.Linear(input_size, 1, bias=True, dtype=torch.float32, device=device)

    def forward(self, hidden: torch.Tensor, markov_embedding: torch.Tensor | None) -> torch.Tensor:
        if self.with_markov:
            if markov_embedding is None:
                raise ValueError("DSpark confidence head requires Markov embeddings")
            hidden = torch.cat((hidden, markov_embedding), dim=-1)
        return torch.sigmoid(self.proj(hidden.float())).squeeze(-1)


@dataclass
class DSparkConfig(Qwen3Config):
    """DSpark head parameters on top of the Qwen3 backbone."""

    markov_rank: int = 0
    enable_confidence_head: bool = False
    confidence_head_with_markov: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> DSparkConfig:
        base = Qwen3Config.from_dict(d)
        return cls(
            **base.__dict__,
            markov_rank=int(d.get("markov_rank") or 0),
            enable_confidence_head=bool(d.get("enable_confidence_head")),
            confidence_head_with_markov=bool(d.get("confidence_head_with_markov")),
        )


class DSparkForCausalLM(PyModelBase):
    """DSpark draft. Registered under ``qwen3_dspark``."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = DSparkConfig.from_dict(config)
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu:0"))
        self.dtype = dtype
        self.device = device
        tp = self.cfg.tp_size
        assert self.cfg.vocab_size % tp == 0

        self.model = DFlashModel(self.cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

        self.markov_head = (
            DSparkMarkovHead(
                self.cfg.vocab_size,
                self.cfg.markov_rank,
                dtype,
                device,
            )
            if self.cfg.markov_rank > 0
            else None
        )

        self.confidence_head = (
            DSparkConfidenceHead(
                self.cfg.hidden_size,
                self.cfg.markov_rank,
                self.cfg.confidence_head_with_markov,
                device,
            )
            if self.cfg.enable_confidence_head
            else None
        )

        # Pybind entry point called from C++ PyCausalLM::write_context_kv.
        self.write_context_kv = self.model.write_context_kv

    # -- Pybind bridge (called from the C++ DSparkWorkerImpl) --------------
    def dspark_markov_bias(self, previous_token_ids: torch.Tensor) -> torch.Tensor:
        assert self.markov_head is not None, "DSpark markov head is not initialized (markov_rank=0?)."
        return self.markov_head.bias(previous_token_ids)

    def has_dspark_confidence_head(self) -> bool:
        return self.confidence_head is not None

    def dspark_confidence_probs(
        self,
        hidden_all: torch.Tensor,
        previous_token_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.confidence_head is not None
        markov_embedding = None
        if previous_token_ids is not None and self.markov_head is not None and self.confidence_head.with_markov:
            markov_embedding = self.markov_head.embed(previous_token_ids)
        return self.confidence_head(hidden_all, markov_embedding)

    # -- weight loading ---------------------------------------------------
    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        cfg = self.cfg
        loader = W8A8WeightLoader(self, state_dicts, tp_size, tp_rank)
        kv_rank, kv_world = cfg.kv_shard(tp_rank, tp_size)

        def shard(name: str, dim: int, kv: bool = False) -> torch.Tensor:
            t = loader.load_tensor(name)
            if kv:
                return loader.shard(t, dim=dim, world=kv_world, rank=kv_rank)
            return loader.shard(t, dim=dim)

        def copy_in(param_name: str, tensor: torch.Tensor) -> None:
            # Fresh lookup (not loader.copy_in): fc is rebuilt below after the
            # loader cached its params, so a cached handle would be stale.
            param = self.get_parameter(param_name)
            param.data.copy_(tensor.to(dtype=param.dtype, device=param.device))

        # embed_tokens / lm_head: the draft owns its own copies.
        copy_in("model.embed_tokens.weight", shard("embed_tokens.weight", dim=1))
        copy_in("lm_head.weight", shard("lm_head.weight", dim=0))

        # Aux-hidden projection. fc's real in-dim
        # (hidden * num_captured_target_layers) is authoritative from the
        # checkpoint; rebuild if the construction-time placeholder disagreed.
        fc_weight = loader.load_tensor("fc.weight")
        if fc_weight.shape[1] != self.model.fc.weight.shape[1]:
            self.model.fc = ColumnParallelLinear(
                fc_weight.shape[1],
                self.cfg.hidden_size,
                1,
                bias=False,
                dtype=self.dtype,
                device=self.device,
            )
        copy_in("model.fc.weight", fc_weight)
        copy_in("model.hidden_norm.weight", loader.load_tensor("hidden_norm.weight"))
        copy_in("model.norm.weight", loader.load_tensor("norm.weight"))

        self.model.load_backbone_layers(
            src_prefix="layers.",
            load_tensor=loader.load_tensor,
            shard=shard,
            copy_in=copy_in,
        )

        # Markov head: markov_w1 = Embedding.weight [vocab, rank];
        # markov_w2 = Linear.weight [vocab, rank].
        if self.markov_head is not None:
            copy_in("markov_head.markov_w1.weight", loader.load_tensor("markov_head.markov_w1.weight"))
            copy_in("markov_head.markov_w2.weight", loader.load_tensor("markov_head.markov_w2.weight"))

        # Confidence head (optional).
        if self.confidence_head is not None:
            copy_in("confidence_head.proj.weight", loader.load_tensor("confidence_head.proj.weight"))
            bias_sd = loader.find("confidence_head.proj.bias")
            if bias_sd is not None:
                copy_in("confidence_head.proj.bias", bias_sd.get_tensor("confidence_head.proj.bias"))
            else:
                self.confidence_head.proj.bias.data.zero_()
