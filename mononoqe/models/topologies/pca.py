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


# This topology must be done with a full dataset size
@register("pca")
def pca_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]
    components = extra_params["components"]

    return [
        topology(
            path=f"output/layers/pca_{components}",
            sequence=[
                layers.flatten(),
                layers.pca(components),
            ],
        ),
        layers.linear(output_size=nb_output_classes),
    ]
