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

from torch.nn import Module

from mononoqe.models.layers.utils import build_topology_from_list, predict_size, register


class Repeat(Module):
    def __init__(self, sequence, iterations):
        super().__init__()
        assert sequence
        assert int(iterations)

        self.sequence = sequence
        self.iterations = iterations

    def forward(self, x):
        for _ in range(0, self.iterations):
            x = self.sequence(x)

        return x


@register
class RepeatBuilder:
    TYPE = "repeat"

    @classmethod
    def elem(cls, sequence: list, iterations: int) -> dict:
        assert int(iterations)
        assert sequence
        return dict(type=cls.TYPE, sequence=sequence, iterations=iterations)

    @classmethod
    def make(cls, input_size, sequence, iterations, **kwargs) -> Module:
        seq, output_size = build_topology_from_list(
            sequence=sequence,
            input_size=input_size,
        )

        return Repeat(seq, iterations)

    @classmethod
    def predict_size(cls, input_size, sequence, **kwargs) -> tuple:
        return predict_size(sequence=sequence, input_size=input_size)
