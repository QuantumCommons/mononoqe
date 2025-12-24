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
class PReLUBuilder:
    TYPE = "prelu"

    @classmethod
    def elem(cls, nb_params: int) -> dict:
        return dict(type=cls.TYPE, nb_params=nb_params)

    @classmethod
    def single(cls) -> dict:
        return cls.elem(1)

    @classmethod
    def multi(cls) -> dict:
        return cls.elem(-1)

    @classmethod
    def make(cls, input_size, nb_params=1, **kwargs) -> Module:
        if nb_params != 1:
            nb_params = input_size

            if isinstance(nb_params, tuple) and len(nb_params) == 3:
                nb_params = nb_params[0]

        return torch.nn.PReLU(num_parameters=nb_params)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size
