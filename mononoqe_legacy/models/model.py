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

from typing import Union
import pytorch_lightning as pl

from mononoqe.data.dataset import Data
from mononoqe.models.topologies.topology import Topology, TopologyParams, build_topology


# class Net(abc.ABC, pl.LightningModule):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     @abc.abstractmethod
#     def configure_training(self, training_params):
#         pass

#     @abc.abstractmethod
#     def configure_topology(self, topology):
#         pass

class Net(pl.LightningModule):

    def __init__(self, data: Data, topology: Union[TopologyParams, Topology], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._data = data

        # Build the topology
        if isinstance(topology, TopologyParams):
            self._topology = build_topology(topology)
        elif isinstance(topology, Topology):
            self._topology = topology
        else:
            raise Exception("Uncompatible type for topology :", type(topology))

        if not self._topology:
            raise Exception("Failed to build topology")

    def forward(self, x):
        raise NotImplementedError()
        
