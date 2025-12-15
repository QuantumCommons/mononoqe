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
class AjaxBuilder:
    TYPE = "ajax"

    @classmethod
    def make(cls, x: torch.Tensor, **kwargs) -> FeatureMapParams:
        x = x.squeeze(0)
        modes_count, photons_count, min_photon = AjaxBuilder.predict_size(x.shape)

        t = x * torch.pi * 2
        circuit = pcvl.Circuit(m=modes_count)

        data_index = 0
        for j in range(modes_count):
            for i in range(modes_count - j - 1):
                angle = float(t[data_index])
                circuit.add(i, pcvl.BS(angle).add(0, pcvl.PS(angle)))
                data_index += 1

        input_state = modes_count * [0]
        places = torch.linspace(0, modes_count - 1, photons_count)

        for photon in places:
            input_state[int(photon)] = 1

        input_state = pcvl.BasicState(input_state)

        return FeatureMapParams(
            circuit=circuit,
            input_state=input_state,
            min_detect_photon=min_photon,
        )

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> Tuple:
        nb_data_to_encode = input_size[0]
        max_modes_count = 30

        # Counting the number of mode required to store all the data in a triangle circuit
        for i in range(max_modes_count):
            nb_of_components = sum(i + 1 for i in range(i))
            if nb_of_components > nb_data_to_encode:
                modes_count = i
                break

        photons_count = 3
        min_photon = photons_count

        # (modes, nb photon, min_photon)
        return (modes_count, photons_count, min_photon)
