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
import os
from typing import Optional

from torch import Generator
from torch.utils.data import DataLoader, Dataset


@dataclass
class Data:
    batch_size: int
    training_dataset: Dataset
    validation_dataset: Dataset
    input_shape: Optional[tuple[int, ...]] = None
    output_shape: Optional[tuple[int, ...]] = None
    device: str = None

    def __post_init__(self):
        if not self.input_shape:
            self.input_shape = tuple(self.training_dataset[0][0].shape)
        if not self.output_shape:
            self.output_shape = tuple(self.training_dataset[0][1].shape)

    def build_loaders(self) -> tuple[DataLoader, DataLoader, tuple[int, ...], tuple[int, ...]]:

        if self.batch_size == -1:
            batch_size = len(self.training_dataset)
        else:
            batch_size = self.batch_size

        num_workers = os.cpu_count() or 0

        train_loader = DataLoader(
            self.training_dataset,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
            generator=Generator(device=self.device),
        )
        val_loader = DataLoader(
            self.validation_dataset,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True,
            generator=Generator(device=self.device),
        )

        return train_loader, val_loader, self.input_shape, self.output_shape


if __name__ == "__main__":

    from pprint import pp

    class DummyDataset(Dataset):
        def __init__(self):
            super().__init__()
            import torch
            self.data = [(torch.randn(3, 224, 224), torch.randint(0, 10, (1,))) for _ in range(42)]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    dummy_dataset = DummyDataset()

    data = Data(
        batch_size=2,
        training_dataset=dummy_dataset,
        validation_dataset=dummy_dataset,
    )

    pp(data.input_shape)
    pp(data.output_shape)

    train_loader, val_loader, input_shape, output_shape = data.build_loaders()
    for i, batch in enumerate(train_loader):
        pp(i)
        pp(batch[0].shape)
