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
from dataclasses_json import dataclass_json
import json
import os
import torch
from torch.nn import Sequential
from typing import Iterable, List, Dict, Tuple, Optional, Union

from mononoqe.models.layers.utils import layers_factory
from mononoqe.models.topologies.register import factory
from mononoqe.utils import make_tuple


@dataclass_json
@dataclass
class TopologyParams:
    name: str
    input_shape: Union[Tuple, int] = field(default=None)
    output_shape: Union[Tuple, int] = field(default=None)
    extra: Optional[Dict] = field(default=None)


@dataclass
class Topology:
    __params_filename = "topology_params.json"
    __seq_list_filename = "topology_list.json"
    __weights_filename = "topology_weights.pt"

    sequence_list: List
    sequence_modules: Sequential
    params: TopologyParams = field(default=None)

    def save(self, path: str):
        with open(os.path.join(path, Topology.__seq_list_filename), "w") as out:
            seq_obj = json.dumps(self.sequence_list, indent=4)
            out.write(seq_obj)

        with open(os.path.join(path, Topology.__params_filename), "w") as out:
            out.write(self.params.to_json())

        torch.save(
            self.sequence_modules.state_dict(),
            os.path.join(path, Topology.__weights_filename),
        )

    @staticmethod
    def load(path: str) -> "Topology":
        if not os.path.isdir(path):
            print(path)
            raise Exception(path, "must be a directory")

        seq_filepath = os.path.join(path, Topology.__seq_list_filename)
        params_filepath = os.path.join(path, Topology.__params_filename)
        weight_filepath = os.path.join(path, Topology.__weights_filename)

        if not os.path.exists(seq_filepath):
            raise Exception(seq_filepath, "doesn't exist")

        if not os.path.exists(params_filepath):
            raise Exception(params_filepath, "doesn't exist")

        if not os.path.exists(weight_filepath):
            raise Exception(weight_filepath, "doesn't exist")

        with open(seq_filepath, "r") as f:
            loaded_topology_list = json.load(f)

        with open(params_filepath, "r") as f:
            loaded_params = TopologyParams.from_json(f.read())

        loaded_weights = torch.load(weight_filepath, weights_only=True)

        topology = build_topology_from_params(loaded_params, loaded_topology_list)
        topology.sequence_modules.load_state_dict(loaded_weights)

        return topology

    def get_weights(self, key: str):
        sd = self.sequence_modules.state_dict()
        if key not in sd.keys():
            raise ValueError(
                f"key {key} is not in topology. available weights: {sd.keys()}"
            )
        return sd[key]


def build_topology_from_params(
    topology_params: TopologyParams, loaded_topology_list: Optional[List] = None
) -> Topology:
    """
    Builds a topology object from TopologyParams.
    If loaded_topology_list is provided, it will be used instead of the factory function.
    This is useful for loading a topology from a registered implementation, mainly for CLI usage.
    """

    if loaded_topology_list:
        topology_pattern_list = loaded_topology_list
    else:
        topology_pattern_fct = factory()[topology_params.name]
        topology_pattern_list = topology_pattern_fct(
            input_shape=topology_params.input_shape,
            output_shape=topology_params.output_shape,
            **topology_params.extra if topology_params.extra else {},
        )

    topology_sequence, built_output_shape = build_topology_from_list(
        topology_pattern_list, topology_params.input_shape
    )

    assert (
        make_tuple(topology_params.output_shape) == built_output_shape
    ), f"Different output size. Expected:{topology_params.output_shape}, Built:{built_output_shape}"

    return Topology(
        params=topology_params,
        sequence_modules=topology_sequence,
        sequence_list=topology_pattern_list,
    )

# def build_topology_from_list(
#     sequence: List, input_size: Tuple = (1,)
# ) -> Tuple[Sequential, Tuple]:
#     if not sequence:
#         return None

#     if isinstance(sequence, dict):
#         sequence = sequence["sequence"]

#     modules = []
#     previous_size = input_size

#     for desc_layer in sequence:
#         layer_class = layers_factory()[desc_layer["type"]]
#         output_size = layer_class.predict_size(input_size=previous_size, **desc_layer)
#         module = layer_class.make(input_size=previous_size, **desc_layer)

#         previous_size = make_tuple(output_size)

#         modules.append(module)

#     return Sequential(*modules), previous_size

def build_topology_from_list(
    sequence: List[Union[str, dict]], input_size: Tuple, name: str = "generated_topology"
) -> Topology:
    """
    Build a topology from a serialized list of layers. Needs an input size too.
    Best used when deserializing a topology.
    """

    if not sequence:
        return None

    if isinstance(sequence, dict):
        sequence = sequence["sequence"]

    modules = []
    previous_size = input_size

    for desc_layer in sequence:
        layer_class = layers_factory()[desc_layer["type"]]
        output_size = layer_class.predict_size(input_size=previous_size, **desc_layer)
        module = layer_class.make(input_size=previous_size, **desc_layer)

        previous_size = make_tuple(output_size)

        modules.append(module)

    params = TopologyParams(
        name=name,
        input_size=input_size,
        output_size=previous_size,
    )

    return Topology(
        params=params,
        sequence_modules=Sequential(*modules),
        sequence_list=sequence,
    )

def build_topology_auto(
    sequence: Union[Iterable[dict], dict],
    input_size: Tuple,
    output_size: Optional[Tuple],
    name: str = "generated_topology",
) -> Topology:
    """
    Builds a topology from a sequence of layers.
    Needs at least the last layer's output size to be specified OR output_size parameter to be provided.
    Will automatically compute all intermediate sizes if not specified.
    """

    if isinstance(sequence, dict):
        sequence = sequence["sequence"]

    if output_size is None:
        last_layer = sequence[-1]
