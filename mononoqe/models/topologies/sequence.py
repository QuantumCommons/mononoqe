# Copyright 2025 Scaleway
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from torch.nn import Module
from typing import List, Optional
import uuid

from mononoqe.models.topologies.topology import Topology, TopologyParams
from mononoqe.models.layers.utils import build_topology_from_list, predict_size, register
from mononoqe.utils import make_tuple


class Sequence(Module):
    def __init__(self, path, topology: Topology, learnable: bool):
        super().__init__()

        assert path
        assert topology

        self.path = path
        self.topology = topology
        self.sequence = topology.sequence_modules
        self.learnable = learnable

        self.sequence.requires_grad_(learnable)

    def forward(self, x):
        return self.sequence(x)

    def __del__(self):
        if not self.learnable:
            return

        from pathlib import Path

        print("Saving topology at", self.path)

        Path(self.path).mkdir(parents=True, exist_ok=True)

        self.topology.save(self.path)


@register
class SequenceBuilder:
    TYPE = "sequence"

    @classmethod
    def elem(
        cls, path: str, sequence: Optional[List] = None, learnable: bool = True
    ) -> dict:
        assert path

        load = False

        if not sequence and os.path.exists(path):
            print("Loading existing topology at", path)
            topology = Topology.load(path)
            sequence = topology.sequence_list
            load = True

        if not isinstance(sequence, List):
            sequence = [sequence]

        return dict(
            type=cls.TYPE, sequence=sequence, id=path, load=load, learnable=learnable
        )

    @classmethod
    def make(cls, input_size, sequence, id, load, learnable, **kwargs) -> Module:
        path = id

        if load:
            topology = Topology.load(path)
        else:
            topology_sequence, built_output_shape = build_topology_from_list(
                sequence, input_size
            )

            topology = Topology(
                params=TopologyParams(
                    name=str(uuid.uuid4()),
                    input_shape=make_tuple(input_size),
                    output_shape=make_tuple(built_output_shape),
                ),
                sequence_list=sequence,
                sequence_modules=topology_sequence,
            )

        return Sequence(path=path, topology=topology, learnable=learnable)

    @classmethod
    def predict_size(cls, input_size, sequence, **kwargs) -> tuple:
        return predict_size(sequence=sequence, input_size=input_size)


topology = SequenceBuilder.elem
