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

import perceval as pcvl
import torch
from typing import Tuple

from mononoqe.models.layers.quantum.ansatz.register import ansatz_factory


def build_ansatz(name: str, x: torch.Tensor, weights: torch.Tensor) -> pcvl.Circuit:
    return ansatz_factory()[name].make(x, weights)


def predict_size(name: str, input_size: torch.Size) -> Tuple:
    return ansatz_factory()[name].predict_size(input_size)


def build_weights(name: str, input_size: torch.Size, **kwargs) -> torch.Tensor:
    return ansatz_factory()[name].make_weights(input_size, **kwargs)
