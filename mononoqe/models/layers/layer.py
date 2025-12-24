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

from torch.nn import Module


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
    def to_dict(self) -> dict:
        """
        Used to serialize the layer.
        Contains every argument needed to build the corresponding torch.nn.Module.
        """
        pass

    @abstractmethod
    def make(self, input_size: int) -> Module:
        """
        Used to generate the torch Module corresponding to the layer.
        """
        pass

    @abstractmethod
    def predict_size(self, input_size: int) -> int:
        """
        Used to predict the size of the output of the layer given the input size.
        """
        pass
