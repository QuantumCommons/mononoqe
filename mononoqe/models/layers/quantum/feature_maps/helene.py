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
class HeleneBuilder:
    TYPE = "helene"

    @classmethod
    def make(cls, x: torch.Tensor, **kwargs) -> FeatureMapParams:
        x = x.squeeze(0)
        modes_count, photons_count, min_photon = HeleneBuilder.predict_size(x.shape)

        circuit = pcvl.Circuit(m=modes_count)

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

        photons_count = 4
        modes_count = nb_data_to_encode
        min_photon = photons_count

        # (modes, nb photon, min_photon)
        return (modes_count, photons_count, min_photon)
