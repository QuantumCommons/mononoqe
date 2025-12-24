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

from mononoqe.models.layers.utils import register


@register
class LeakyReLUBuilder:
    TYPE = "lrelu"

    @classmethod
    def elem(cls, alpha: float = 0.01, inplace: bool = False) -> dict:
        return dict(type=cls.TYPE, alpha=alpha, inplace=inplace)

    @classmethod
    def make(cls, alpha, inplace, **kwargs) -> Module:
        return torch.nn.LeakyReLU(alpha, inplace)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size
