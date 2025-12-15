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

from mononoqe.models.layers.quantum.feature_maps.register import register
from mononoqe.models.layers.quantum.feature_maps.feature_map_params import FeatureMapParams


@register
class HectorBuilder:
    TYPE = "hector"
    # Best mode indexation is based on circuit analysis to send PCA feature into most probable output
    INDEXES_12 = torch.Tensor([1, 10, 9, 2, 5, 6, 11, 0, 7, 4, 8, 3]).to(torch.int32)

    @classmethod
    def make(cls, x: torch.Tensor, **kwargs) -> FeatureMapParams:
        x = x.squeeze(0)
        modes_count, _, min_photon = HectorBuilder.predict_size(x.shape)

        circuit = pcvl.Circuit(m=modes_count)

        angle = torch.pi / 2

        for i in range(modes_count // 2 - 1):
            circuit.add(i, pcvl.BS(theta=angle))

        for i in reversed(range(modes_count // 2, modes_count - 1)):
            circuit.add(i, pcvl.BS(theta=angle))

        circuit.add(modes_count // 2 - 1, pcvl.BS(theta=angle))

        for i in reversed(range(modes_count // 2 - 1)):
            circuit.add(i, pcvl.BS(theta=angle))

        for i in range(modes_count // 2, modes_count - 1):
            circuit.add(i, pcvl.BS(theta=angle))

        # Feature map
        x_arr = x[HectorBuilder.INDEXES_12]
        x_normalized = (x_arr / torch.max(torch.abs(x_arr))) * torch.pi / 2

        for i in range(modes_count):
            p_val = float(x_normalized[i])
            circuit.add(i, pcvl.PS(p_val))

        input_state = pcvl.BasicState("|0,0,1,1,0,0,0,0,1,1,0,0>")

        return FeatureMapParams(
            circuit=circuit,
            input_state=input_state,
            min_detect_photon=min_photon,
        )

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> Tuple:
        nb_data_to_encode = input_size[0]

        photons_count = 4
        modes_count = nb_data_to_encode
        min_photon = photons_count

        # (modes, nb photon, min_photon)
        return (modes_count, photons_count, min_photon)
