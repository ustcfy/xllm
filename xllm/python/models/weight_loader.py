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

"""Architecture-neutral checkpoint weight loader shared by model load_weights.

Owns the byte-identical state-dict lookup, TP sharding, and param/buffer copy
mechanics common to every model's ``load_weights``. Quantization- or
architecture-specific packing (e.g. W8A8) extends this base with extra methods;
the model-specific per-layer loop stays in each model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


class WeightLoader:
    """Generic checkpoint tensor lookup / TP sharding / param copy."""

    def __init__(
        self,
        model: nn.Module,
        state_dicts: list[StateDict],
        tp_size: int,
        tp_rank: int,
        kv_world: Optional[int] = None,
        kv_rank: Optional[int] = None,
    ) -> None:
        self._model = model
        self._state_dicts = state_dicts
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        # KV projections may shard over a smaller (GQA) group than TP.
        self.kv_world = tp_size if kv_world is None else kv_world
        self.kv_rank = tp_rank if kv_rank is None else kv_rank

    def find(self, name: str) -> Optional[StateDict]:
        for sd in self._state_dicts:
            if sd.has(name):
                return sd
        return None

    def load_tensor(self, name: str) -> torch.Tensor:
        sd = self.find(name)
        assert sd is not None, f"checkpoint tensor not found: {name}"
        return sd.get_tensor(name)

    def load_shard(self, name: str, dim: int, kv: bool = False) -> torch.Tensor:
        """Load a checkpoint tensor and shard it along ``dim``, over the KV
        group when ``kv`` else the TP group.
        """
        t = self.load_tensor(name)
        if kv:
            return self.shard(t, dim=dim, world=self.kv_world, rank=self.kv_rank)
        return self.shard(t, dim=dim)

    def shard(
        self,
        t: torch.Tensor,
        dim: int,
        world: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> torch.Tensor:
        world = self.tp_size if world is None else world
        rank = self.tp_rank if rank is None else rank
        if world <= 1:
            return t
        cs = t.size(dim) // world
        return t.narrow(dim, rank * cs, cs).contiguous()

    def copy_in(self, param_name: str, tensor: torch.Tensor) -> None:
        # Resolve live, not from a snapshot: callers may rebuild a submodule
        # after constructing the loader (e.g. the DSpark draft's fc).
        try:
            p = self._model.get_parameter(param_name)
        except AttributeError:
            p = self._model.get_buffer(param_name)
        p.data.copy_(tensor.to(dtype=p.dtype, device=p.device))
