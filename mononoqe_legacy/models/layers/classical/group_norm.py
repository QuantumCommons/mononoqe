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
class GroupNormBuilder:
    TYPE = "group_norm"

    @classmethod
    def elem(cls, num_groups: int, num_channels: int) -> dict:
        return dict(type=cls.TYPE, num_groups=num_groups, num_channels=num_channels)

    @classmethod
    def make(cls, input_size, num_groups, num_channels, **kwargs) -> Module:
        return torch.nn.GroupNorm(num_groups, num_channels)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size
