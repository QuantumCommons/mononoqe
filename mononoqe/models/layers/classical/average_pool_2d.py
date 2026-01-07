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

from dataclasses import field
import math

import torch

from mononoqe.models.layers import register, Layer

@register
class AveragePool2D(Layer):
    name = "average_pool_2d"

    kernel_size: int | tuple = field(default=3)
    stride: int | tuple = field(default=None)
    padding: int | tuple = field(default=0)
    ceil_mode: bool = field(default=False)
    count_include_pad: bool = field(default=True)
    divisor_override: int = field(default=None)

    def make(self, input_shape: tuple) -> torch.nn.Module:
        return torch.nn.AvgPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )

    def predict_shape(self, input_shape: tuple) -> tuple:
        output_h = math.floor(
            (input_shape[1] + 2 * self.padding[0] - self.kernel[0]) / self.stride[0] + 1
        )
        output_w = math.floor(
            (input_shape[2] + 2 * self.padding[1] - self.kernel[1]) / self.stride[1] + 1
        )

        return (input_shape[0], output_h, output_w)
