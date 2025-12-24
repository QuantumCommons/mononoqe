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

from mononoqe.models.layers.quantum.gradients.register import register


@register("dummy")
def dummy_method(ctx, grad_output: torch.Tensor):
    _, weights = ctx.saved_tensors

    grad_weights = torch.full_like(weights, torch.mean(grad_output))

    # grad_input, grad_weight, fw, bw
    return None, grad_weights, None, None
