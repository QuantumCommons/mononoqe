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

import torch


class LayerBuilder(ABC):
    """Abstract base class for building neural network layers."""

    @property
    @abstractmethod
    def TYPE() -> str:
        return "abstract"

    @classmethod
    @abstractmethod
    def elem(cls, *args, **kwargs) -> dict:
        return dict(type=cls.TYPE, *args, **kwargs)

    @classmethod
    @abstractmethod
    def make(cls, *args, **kwargs) -> torch.nn.Module:
        pass

    @classmethod
    @abstractmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        pass


if __name__ == "__main__":

    from pprint import pp

    pp("Builder module")

    class TestLayerBuilder(LayerBuilder):
        TYPE = "test"

        def elem(cls, *args, **kwargs):
            return super().elem(*args, **kwargs)

        def make(cls, *args, **kwargs):
            return torch.nn.Linear(1, 1)
        
        def predict_size(cls, input_size, **kwargs):
            return input_size

    pp(TestLayerBuilder)

    test_builder = TestLayerBuilder()

    pp(test_builder)

    pp(test_builder.elem())
    pp(test_builder.make())
    pp(test_builder.predict_size((1, 1)))

    pp("Builder module end")
