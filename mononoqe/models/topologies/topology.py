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


class Topology(ABC):
    """
    Used to agregate a collection of layers into model topology.
    It shall be able to accept different kind of inputs (registered topology names, layer lists, serialized dictionnary of layers, etc.),
    and build the appropriate torch.nn.Module sequence from it.
    It shall be able to infer the input and/or output size of layers when not provided, using each layer's default size attributes.
    Of course, it at least need to be given the input size of the first layer and output size of the last layer. It may be given using the 'Data' class (see mononoqe/data/data.py)
    """
    ...