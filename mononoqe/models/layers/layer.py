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

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from mononoqe.models.layers import LayerType
from mononoqe.utils import Factory


__FACTORY = Factory("layer")

def factory() -> Factory:
    global __FACTORY
    return __FACTORY

def register(name: str):
    return factory().register(name)


@dataclass
class Layer(ABC):
    """
    This is the interface for all layers.
    It is the building block of Topologies.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Used to identify the layer.
        """
        pass

    @abstractmethod
    def make(self, input_shape: tuple) -> torch.nn.Module:
        """
        Used to generate the torch Module corresponding to the layer.
        """
        pass

    @abstractmethod
    def predict_shape(self, input_shape: tuple) -> tuple:
        """
        Used to predict the size of the output of the layer given the input size.
        """
        pass

    def __call__(self, input_shape: tuple, *args, **kwargs) -> torch.nn.Module:
        return self.make(*args, **kwargs)


def predict_sequence_shape(sequence: list[LayerType], input_shape: tuple) -> tuple[int]:
    if not sequence:
        return input_shape

    if isinstance(sequence, dict):
        sequence = sequence["sequence"]

    layers_factory = factory()

    previous_size = input_shape

    for desc_layer in sequence:
        if isinstance(desc_layer, str):
            desc_layer = layers_factory[desc_layer]
        previous_size = desc_layer.predict_shape(input_shape=previous_size)

    return previous_size


