# Copyright 2026 Scaleway
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
from typing import Callable, Iterator, Optional, Union

import torch

from mononoqe.models.hyperparameters import build_loss, build_optimizer, build_scheduler


@dataclass_json
@dataclass
class Hyperparameters:
    loss: Union[str, Callable]
    optimizer: Union[str, torch.optim.Optimizer]
    scheduler: Union[str, torch.optim.lr_scheduler.LRScheduler] = field(default=None)
    learning_rate: float = field(default=3e-4)  # Karpathy constant
    epochs: int = field(default=5)

    def build_minimizers(self, parameters: Iterator[torch.nn.Parameter]) -> tuple[Callable[[torch.Tensor, torch.Tensor, Optional[dict]], torch.Tensor], torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:

        loss = self.loss
        if isinstance(self.loss, str):
            loss = build_loss(self.loss_name)
            self.loss = loss

        optimizer = self.optimizer
        if isinstance(self.optimizer, str):
            optimizer = build_optimizer(
                self.optimizer_name,
                parameters,
                {"lr": self.learning_rate},
            )
            self.optimizer = optimizer

        scheduler = self.scheduler
        if isinstance(self.scheduler, str):
            scheduler = build_scheduler(self.scheduler, optimizer)
            self.scheduler = scheduler

        return loss, optimizer, scheduler
