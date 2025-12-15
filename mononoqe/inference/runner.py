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

import torch
import pytorch_lightning as pl

from mononoqe.data import ValidationData
from mononoqe.models import Net


class Validator:
    def __init__(self) -> "Validator":
        pass

    def predict(
        self,
        model: Net,
        validation_data: ValidationData,
    ):
        trainer = pl.Trainer()

        validation_dataloader, _, _ = validation_data.build_loaders()

        model.train(False)
        model.requires_grad_(False)

        trainer.validate(model, validation_dataloader)


class Runner:
    def __init__(self) -> "Runner":
        pass

    # TODO: implem custom runner on single image/tensor
    def predict(self, model: Net, input: torch.Tensor):
        pass
