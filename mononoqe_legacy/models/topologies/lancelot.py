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


# Work in progress, doesn't converge
@register("lancelot_autoencoder")
def lancelot_autoencoder_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    return [
        topology(
            path="output/layers/lancelot_latent_space",
            sequence=[
                layers.flatten(),
                layers.linear(18),
                layers.prelu(),
                layers.linear(16),
                layers.prelu(),
            ],
        ),
        layers.reshape(shape=(1, 4, 4)),
        layers.conv_2d(kernel=2, output_chan=32),
        layers.prelu(),
        layers.pixelshuffle(factor=2),
        layers.conv_2d(kernel=2, output_chan=16),
        layers.prelu(),
        layers.pixelshuffle(factor=2),
        layers.conv_2d(kernel=2, output_chan=8),
        layers.prelu(),
        layers.pixelshuffle(factor=2),
        layers.conv_2d(kernel=3, output_chan=4),
        layers.prelu(),
        layers.pixelshuffle(factor=2),
        layers.conv_2d(kernel=5, output_chan=1),
    ]


# Work in progress, doesn't converge
@register("lancelot_classifier")
def lancelot_classifier_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]

    return [
        topology("output/layers/lancelot_latent_space", learnable=False),
        layers.linear(25),
        layers.mish(),
        layers.linear(output_size=nb_output_classes),
    ]
