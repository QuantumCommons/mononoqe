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

from inspect import signature
import os
from typing import Optional

from torch import Generator
from torch.utils.data import DataLoader, Dataset


class Data:

    def __init__(
        self,
        batch_size: int,
        training_dataset: Dataset,
        validation_dataset: Dataset,
        input_shape: Optional[tuple[int, ...]] = None,
        output_shape: Optional[tuple[int, ...]] = None,
        device: str = None,
        **kwargs,
    ):
        """
        Use kwargs to pass any additional arguments to the dataloader constructor.
        """
        self._training_dataset = training_dataset
        self._validation_dataset = validation_dataset
        self._batch_size = len(training_dataset) if batch_size <= 0 else batch_size
        self._device = device
        self._input_shape = input_shape
        self._output_shape = output_shape

        if not self._input_shape:
            self._input_shape = tuple(self._training_dataset[0][0].shape)
        if not self._output_shape:
            self._output_shape = tuple(self._training_dataset[0][1].shape)

        self._dataloader_options = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': os.cpu_count() or 0,
            'persistent_workers': True,
            'generator': Generator(device=device) if device else None,
        }
        self._dataloader_options.update({
            key: value
            for key, value in kwargs.items()
            if key in signature(DataLoader.__init__).parameters
        })

    def build_loaders(
        self,
    ) -> tuple[DataLoader, DataLoader, tuple[int, ...], tuple[int, ...]]:

        train_loader = DataLoader(
            self._training_dataset,
            **self._dataloader_options,
        )
        val_loader = DataLoader(
            self._validation_dataset,
            **self._dataloader_options,
        )

        return train_loader, val_loader, self._input_shape, self._output_shape

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def dataloader_options(self) -> dict:
        return self._dataloader_options


if __name__ == "__main__":

    from pprint import pp

    class DummyDataset(Dataset):
        def __init__(self):
            super().__init__()
            import torch

            self.data = [
                (torch.randn(3, 224, 224), torch.randint(0, 10, (1,)))
                for _ in range(42)
            ]

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

    train_loader, val_loader, input_shape, output_shape = data.build_loaders()

    pp(input_shape)
    pp(output_shape)

    for i, batch in enumerate(train_loader):
        pp(i)
        pp(batch[0].shape)
