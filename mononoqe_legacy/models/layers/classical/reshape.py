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

from mononoqe.models.layers.utils import register


class Reshape(Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape if not isinstance(shape, int) else (shape,)

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

    def extra_repr(self) -> str:
        return f"shape={self.shape}"


@register
class ReshapeBuilder:
    TYPE = "reshape"

    @classmethod
    def elem(cls, shape: tuple) -> dict:
        return dict(type=cls.TYPE, output_size=shape)

    @classmethod
    def make(cls, output_size, **kwargs) -> Module:
        return Reshape(output_size)

    @classmethod
    def predict_size(cls, input_size, output_size, **kwargs) -> tuple:
        # Check dimensions match
        if isinstance(input_size, int):
            input_size = (input_size,)
        in_numel = 1
        for dim in input_size:
            in_numel *= dim
        if isinstance(output_size, int):
            output_size = (output_size,)
        out_numel = 1
        for dim in output_size:
            out_numel *= dim
        assert (
            out_numel == in_numel
        ), f"Reshape dimensions are not matching: {in_numel} != {out_numel}"
        # Direct output size
        return output_size
