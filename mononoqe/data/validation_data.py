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

from dataclasses import dataclass
from torch import Generator
from torch.utils.data import DataLoader

from mononoqe.data.dataset import (
    get_validation_mnist_classification_dataset,
    get_validation_mnist_mirror_dataset,
)


@dataclass
class ValidationData:
    batch_size: int
    name: str
    device: str = None

    def build_loaders(self):
        mapping = {
            "mnist_classification": get_validation_mnist_classification_dataset,
            "mnist_mirror": get_validation_mnist_mirror_dataset,
        }

        validation_dataset, input_shape, output_shape = mapping[self.name]()

        val_loader = DataLoader(
            validation_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=6,
            persistent_workers=True,
            generator=Generator(device=self.device),
        )

        return val_loader, input_shape, output_shape
