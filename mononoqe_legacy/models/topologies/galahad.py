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
from typing import List

import mononoqe.models.layers as layers
from mononoqe.models.topologies.register import register
from mononoqe.models.topologies.sequence import topology


@register("galahad")
def galahad_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]
    components = extra_params.get("components", 10)

    return [
        layers.concat(
            sequences=[
                [
                    topology(f"output/layers/pca_{components}", learnable=False),
                    layers.fourier_features_1d(components * 2, 0.1),
                ],
                [
                    topology(f"output/layers/pca_{components}", learnable=False),
                ],
            ]
        ),
        layers.mish(),
        layers.linear(output_size=nb_output_classes),
    ]


@register("galahad_qt")
def galahad_qt_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]
    components = extra_params.get("components", 10)

    return [
        layers.concat(
            sequences=[
                [
                    topology(f"output/layers/pca_{components}", learnable=False),
                    layers.fourier_features_1d(components * 2, 0.1),
                    layers.photonic_circuit(
                        fmap="ajax",
                        converter="cumulative_mode",
                        thresholded_output=False,
                    ),
                ],
                [
                    topology(f"output/layers/pca_{components}", learnable=False),
                ],
            ]
        ),
        layers.mish(),
        layers.linear(output_size=nb_output_classes),
    ]
