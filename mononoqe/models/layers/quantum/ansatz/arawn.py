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

from mononoqe.models.layers.quantum.ansatz.register import register


@register
class ArawnBuilder:
    TYPE = "arawn"

    @classmethod
    def make(cls, x: torch.Tensor, weights: torch.Tensor) -> pcvl.Circuit:
        x = x.squeeze(0)  # batch size = 1
        modes_count = x.shape[0]
        circuit = pcvl.Circuit(modes_count)

        j = 0
        for i in range(modes_count - 1):
            t_val = float(weights[j])
            circuit.add(i, pcvl.BS.Rx(theta=t_val))
            circuit.add(i, pcvl.PS(phi=t_val))
            j += 1

        for i in range(modes_count - 1):
            t_val = float(weights[j])
            circuit.add(i, pcvl.BS.H(theta=t_val))
            circuit.add(i, pcvl.PS(phi=t_val))
            j += 1

        for i in range(modes_count - 1):
            t_val = float(weights[j])
            circuit.add(i, pcvl.BS.Ry(theta=t_val))
            circuit.add(i, pcvl.PS(phi=t_val))
            j += 1

        return circuit

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> Tuple:
        # Ansatz always follows a feature_map, just return the same size
        # (Mode, nb photon, min_photon)
        return input_size

    @classmethod
    def make_weights(cls, input_size, iterations, **kwargs) -> torch.Tensor:
        nb_parameters = input_size[0] * 3
        return torch.FloatTensor(size=(iterations, nb_parameters)).uniform_(
            -torch.pi, torch.pi
        )
