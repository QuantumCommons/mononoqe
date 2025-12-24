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

import math
import torch
from torch.nn import Module
from typing import Union

from mononoqe.utils import make_2d_tuple
from mononoqe.models.layers.utils import register


@register
class MaxPool2dBuilder:
    TYPE = "maxpool2d"

    @classmethod
    def elem(
        cls,
        kernel: Union[int, tuple],
        stride: Union[int, tuple] = (1, 1),
        dilation: Union[int, tuple] = (1, 1),
        padding: Union[int, tuple] = (0, 0),
        ceil_mode: bool = False,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=make_2d_tuple(kernel),
            stride=make_2d_tuple(stride),
            dilation=make_2d_tuple(dilation),
            padding=make_2d_tuple(padding),
            ceil_mode=ceil_mode,
        )

    @classmethod
    def make(
        cls,
        stride,
        dilation,
        padding,
        kernel,
        ceil_mode,
        **kwargs,
    ) -> Module:
        module = torch.nn.MaxPool2d(
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=False,
        )

        return module

    @classmethod
    def predict_size(
        cls, input_size, stride, dilation, padding, kernel, **kwargs
    ) -> tuple:
        output_h = math.floor(
            (input_size[1] + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1)
            / stride[0]
            + 1
        )
        output_w = math.floor(
            (input_size[2] + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1)
            / stride[1]
            + 1
        )

        return (input_size[0], output_h, output_w)
