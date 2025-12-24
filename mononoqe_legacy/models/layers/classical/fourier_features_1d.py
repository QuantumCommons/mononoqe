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

from mononoqe.utils import single_to_int
from mononoqe.models.layers.utils import register


def fourier_features_1d(x, weight):
    vp = 2 * torch.pi * x @ weight.T

    return torch.cat((torch.sin(vp), torch.cos(vp)), dim=-1)


class FourierFeatures1d(Module):
    def __init__(self, input_chan: int, output_chan: int, sigma: float):
        super().__init__()

        assert output_chan % 2 == 0

        self._input_chan = input_chan
        self._mapping_size = output_chan // 2
        self.weight = torch.nn.Parameter(
            torch.randn((self._input_chan, self._mapping_size)) * sigma
        )

    def forward(self, x):
        return fourier_features_1d(x, self.weight.detach())


@register
class FourierFeatures1dBuilder:
    TYPE = "fourier_features_1d"

    @classmethod
    def elem(cls, output_chan: int, sigma: float = 0.1) -> dict:
        return dict(type=cls.TYPE, output_chan=output_chan, sigma=sigma)

    @classmethod
    def make(cls, input_size, output_chan: int, sigma: float, **kwargs) -> Module:
        return FourierFeatures1d(single_to_int(input_size), output_chan, sigma)

    @classmethod
    def predict_size(cls, input_size, output_chan, **kwargs) -> tuple:
        return (output_chan,)
