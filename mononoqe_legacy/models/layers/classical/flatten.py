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

from functools import reduce
from operator import mul
import torch
from torch.nn import Module

from mononoqe.models.layers.utils import register


class Flatten(Module):
    def __init__(self, start_dim: int, end_dim: int):
        super().__init__()
        assert start_dim >= 0, "dim has to be positive or None"

        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return torch.flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)


@register
class FlattenBuilder:
    TYPE = "flatten"

    @classmethod
    def elem(cls, start_dim: int = 0, end_dim: int = -1) -> dict:
        return dict(type=cls.TYPE, start_dim=start_dim, end_dim=end_dim)

    @classmethod
    def make(cls, start_dim: int, end_dim: int, **kwargs) -> Module:
        start_dim = start_dim + 1  # shift because of batch at dim 0
        end_dim = (
            end_dim if end_dim == -1 else end_dim + 1
        )  # shift because of batch at dim 0

        return Flatten(start_dim=start_dim, end_dim=end_dim)

    @classmethod
    def predict_size(cls, input_size, start_dim: int, end_dim: int, **kwargs) -> tuple:
        if end_dim != -1:
            select = input_size[start_dim:end_dim]
        else:
            select = input_size[start_dim:]

        size = reduce(mul, select)  # multiply all elements

        return (size,)
