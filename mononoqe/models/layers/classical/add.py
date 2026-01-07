# Copyright 2026 Scaleway
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

from typing import Iterable
from torch.nn import Module

from mononoqe.models.layers import Layer, LayerType, predict_sequence_shape, register
from mononoqe.models.topologies import Topology


class AddModule(Module):
    def __init__(self, sequences: Iterable[Module]):
        super().__init__()
        assert sequences, "sequences has to be a non empty list"

        self.sequences = sequences

        for idx, module in enumerate(sequences):
            self.add_module(name=str(idx), module=module)

    def forward(self, x):
        res = self.sequences[0].forward(x)
        for sequence in self.sequences[1:]:
            res = res + sequence.forward(x)
        return res


@register
class Add(Layer):
    name = "add"

    sequences: list[list[LayerType]]

    def make(self, input_shape: tuple) -> Module:
        assert self.sequences, "sequences has to be a non empty list"

        built_sequences: list[Module] = []
        for sequence in self.sequences:

            topology = Topology(sequence)
            built_sequence: Module = topology.build_sequence(input_shape=input_shape)
            built_sequences.append(built_sequence)

        # output shape is supposed to be the same for each sub sequence
        return AddModule(built_sequences)

    def predict_shape(self, input_shape: tuple) -> tuple:
        assert self.sequences, "sequences has to be a non empty list"
        return predict_sequence_shape(sequence=self.sequences[0], input_shape=input_shape)
