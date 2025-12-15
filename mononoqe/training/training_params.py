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
from typing import Dict

from mononoqe.training.optimizers import build_optimizer
from mononoqe.training.losses import build_loss
from mononoqe.training.schedulers import build_scheduler


@dataclass
class TrainingParams:
    loss_name: str
    optimizer_name: str
    scheduler_name: str
    epochs: int
    learning_rate: float

    def build_minimizers(self, model_parameters: Dict):
        loss = build_loss(self.loss_name)

        optimizer = build_optimizer(
            self.optimizer_name,
            model_parameters,
            {"lr": self.learning_rate},
        )

        if self.scheduler_name:
            scheduler = build_scheduler(self.scheduler_name, optimizer)
        else:
            scheduler = None

        return loss, optimizer, scheduler
