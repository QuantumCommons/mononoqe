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
class PixelShuffleBuilder:
    TYPE = "pixel_shuffle"

    @classmethod
    def elem(cls, factor: int) -> dict:
        return dict(type=cls.TYPE, factor=factor)

    @classmethod
    def make(cls, factor, **kwargs) -> Module:
        return torch.nn.PixelShuffle(factor)

    @classmethod
    def predict_size(cls, input_size, factor, **kwargs) -> tuple:
        c_in, h_in, w_in = input_size[-3:]

        ouput_size = tuple(
            [
                *input_size[:-3],
                int(c_in / factor / factor),
                h_in * factor,
                w_in * factor,
            ]
        )

        return ouput_size
