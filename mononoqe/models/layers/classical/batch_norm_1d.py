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

from mononoqe.models.layers import register, Layer


@register
class BatchNorm1d(Layer):
    name = "batchnorm1d"

    eps: float = 0.00001
    momentum: float | None = 0.1
    affine: bool = True
    track_running_stats: bool = True

    def make(self, input_shape: tuple) -> torch.nn.Module:
        return torch.nn.BatchNorm1d(
            input_shape[0],
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )

    def predict_shape(self, input_shape: tuple) -> tuple:
        return input_shape
