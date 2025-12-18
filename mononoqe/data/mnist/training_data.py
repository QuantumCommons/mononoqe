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
from typing import Literal

from mononoqe.data.dataset import (
    get_validation_mnist_classification_dataset,
    get_full_validation_mnist_classification_dataset,
    get_validation_mnist_mirror_dataset,
    get_partial_mnist_classification_dataset,
    get_full_mnist_classification_dataset,
    get_partial_mnist_mirror_dataset,
)


@dataclass
class TrainingData:
    batch_size: int
    name: Literal["partial, full, mirror"] = "full"
    device: str = None

    def build_loaders(self):
        mapping = {
            "partial": (
                get_partial_mnist_classification_dataset,
                get_validation_mnist_classification_dataset,
            ),
            "full": (
                get_full_mnist_classification_dataset,
                get_full_validation_mnist_classification_dataset,
            ),
            "mirror": (
                get_partial_mnist_mirror_dataset,
                get_validation_mnist_mirror_dataset,
            ),
        }

        training_callback, validation_callback = mapping[self.name]
        training_dataset, input_shape, output_shape = training_callback()
        validation_dataset, _, _ = validation_callback()

        if self.batch_size == -1:
            batch_size = len(training_dataset)
        else:
            batch_size = self.batch_size

        train_loader = DataLoader(
            training_dataset,
            batch_size,
            shuffle=True,
            num_workers=6,
            persistent_workers=True,
            generator=Generator(device=self.device),
        )
        val_loader = DataLoader(
            validation_dataset,
            batch_size,
            shuffle=False,
            num_workers=6,
            persistent_workers=True,
            generator=Generator(device=self.device),
        )

        return train_loader, val_loader, input_shape, output_shape
