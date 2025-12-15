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
from typing import Union

from mononoqe.models.layers.utils import register
from mononoqe.utils import make_2d_tuple


@register
class Deconv2dBuilder:
    TYPE = "deconv2d"

    @classmethod
    def elem(
        cls,
        kernel: Union[int, tuple],
        output_chan: Union[int, str] = 1,
        stride: Union[int, tuple] = (1, 1),
        dilation: Union[int, tuple] = (1, 1),
        padding: Union[int, tuple] = (0, 0),
        output_padding: Union[int, tuple] = (0, 0),
        padding_mode: str = "zeros",
        bias: bool = True,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=make_2d_tuple(kernel),
            stride=make_2d_tuple(stride),
            dilation=make_2d_tuple(dilation),
            padding=make_2d_tuple(padding),
            padding_mode=padding_mode,
            output_channels=output_chan,
            output_padding=make_2d_tuple(output_padding),
            bias=bias,
        )

    @classmethod
    def make(
        cls,
        input_size,
        stride,
        dilation,
        padding,
        padding_mode,
        bias,
        kernel,
        output_channels,
        **kwargs,
    ) -> Module:
        input_channel = input_size[0]

        module = torch.nn.ConvTranspose2d(
            in_channels=input_channel,
            out_channels=output_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
        )

        return module

    @classmethod
    def predict_size(
        cls,
        input_size,
        stride,
        dilation,
        padding,
        kernel,
        output_channels,
        output_padding,
        **kwargs,
    ) -> tuple:
        output_h = (
            (input_size[1] - 1) * stride[0]
            - 2 * padding[0]
            + dilation[0] * (kernel[0] - 1)
            + output_padding[0]
            + 1
        )
        output_w = (
            (input_size[2] - 1) * stride[1]
            - 2 * padding[1]
            + dilation[1] * (kernel[1] - 1)
            + output_padding[1]
            + 1
        )

        return (output_channels, output_h, output_w)
