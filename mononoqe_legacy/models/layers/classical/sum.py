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

import torch
from torch.nn import Module
from typing import Optional

from mononoqe.models.layers.utils import register


class Sum(Module):
    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        assert dim >= 0, "dim has to be positive or None"

        self.dim = dim

    def forward(self, x):
        return torch.sum(x, dim=self.dim)


@register
class SumBuilder:
    TYPE = "sum"

    @classmethod
    def elem(cls, dim: Optional[int] = None) -> dict:
        return dict(type=cls.TYPE, dim=dim)

    @classmethod
    def make(cls, dim, **kwargs) -> Module:
        dim = None if dim is None else dim + 1
        return Sum(dim=dim)

    @classmethod
    def predict_size(cls, input_size, sequences, **kwargs) -> tuple:
        return input_size
