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

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Optional, Union

import torch

from mononoqe.models.layers import LayerType, Layer


@dataclass
class Topology():
    """
    Used to agregate a collection of layers into model topology.
    It shall be able to accept different kind of inputs (registered topology names, layer lists, serialized dictionnary of layers, etc.),
    and build the appropriate torch.nn.Module sequence from it.
    It shall be able to infer the input and/or output size of layers when not provided, using each layer's default size attributes.
    Of course, it at least need to be given the input size of the first layer and output size of the last layer. It may be given using the 'Data' class (see mononoqe/data/data.py)
    """

    __seq_list_filename = "topology_list.json"
    __weights_filename = "topology_weights.pt"

    layers: Iterable[LayerType]
    module_sequence: Optional[torch.nn.Sequential] = field(default=None)

    def build_sequence(self, input_shape: tuple, output_shape: Optional[tuple]) -> torch.nn.Sequential:
        """
        Build the sequence of layers from the topology definition.
        The input size is used to initialize the first layer.
        The output size is used as output size of the last layer.
        If output_shape is None, it will be inferred from the last layer's default or defined output size.
        """
        if self.module_sequence is not None:
            return self.module_sequence

        if not self.layers:
            raise ValueError("No layers defined in the topology")

        previous_size = input_shape
        modules = []

        for layer in self.layers:
            if isinstance(layer, Layer):
                module = layer.make(previous_size)
                previous_size = layer.predict_shape(previous_size)
            # elif isinstance(layer, dict):
            #     layer = Layer.from_dict(layer)
            # elif isinstance(layer, str):
            #     layer = Layer.from_str(layer)
            else:
                raise NotImplementedError(f"Layer definition not implemented for type {type(layer)}")
                # raise ValueError(f"Invalid layer definition: {layer}")

            modules.append(module)

        self.module_sequence = torch.nn.Sequential(*modules)
        return self.module_sequence

    def save(self, path: Union[str, Path]):
        """
        Save the topology to a file.
        """
        if not isinstance(path, Path):
            path = Path(path)

        with open(path / Topology.__seq_list_filename, "w") as out:
            seq_obj = json.dumps(self.layers, indent=4)
            out.write(seq_obj)

        torch.save(
            self.module_sequence.state_dict(),
            path / Topology.__weights_filename,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Topology":
        """
        Load a topology from a file.
        """
        if not isinstance(path, Path):
            path = Path(path)

        if not path.is_dir():
            raise Exception(path, "must be a directory")

        seq_filepath = path / Topology.__seq_list_filename
        weight_filepath = path / Topology.__weights_filename

        if not seq_filepath.exists():
            raise Exception(seq_filepath, "doesn't exist")

        if not weight_filepath.exists():
            raise Exception(weight_filepath, "doesn't exist")

        with open(seq_filepath, "r") as f:
            loaded_topology_list = json.load(f)

        loaded_weights = torch.load(weight_filepath, weights_only=True)

        topology = Topology(loaded_topology_list)
        sequence = torch.nn.Sequential()
        sequence.load_state_dict(loaded_weights)
        topology.module_sequence = sequence

        return topology
