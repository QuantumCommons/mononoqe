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

import os

import torch
import pytorch_lightning as pl

from mononoqe.models.topologies import Topology


class Net(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        raise NotImplementedError("Misses attributes init, as well as minimizer's build method.")

        self.__loss = None
        self.__accuracy = None
        self.__topology = None
        self.__sequence = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__sequence(x)

    def training_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        x, y_ref = batch

        y_pred = self.forward(x)

        loss = self.__loss(y_pred, y_ref)
        accuracy = self.__accuracy(y_pred, y_ref)

        self.log(
            "train_accuracy",
            accuracy,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx):
        x, y_ref = batch

        y_pred = self.forward(x)
        accuracy = self.__accuracy(y_pred, y_ref)

        self.log(
            "val_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Can be None in inference mode
        if self.__loss:
            loss = self.__loss(y_pred, y_ref)
            self.log(
                "val_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            return loss

        return accuracy

    ### Used by pytorch_lightning
    def configure_optimizers(self):
        if not self.__training_params:
            return None

        # need to pass model.parameters() to create optimizer object
        loss, optimizer, scheduler = self.__training_params.build_minimizers(
            self.parameters()
        )

        self.__loss = loss

        if not scheduler:
            return optimizer

        return [optimizer], [scheduler]
    ###

    #### Used by pytorch_lightning
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)
    ###

    def save(self, path: str):
        from pathlib import Path

        print("Saving model at", path)

        Path(path).mkdir(parents=True, exist_ok=True)

        self.__topology.save(path)

    @staticmethod
    def load(path: str) -> "Net":
        print("Loading model from", path)

        if not os.path.exists(path):
            raise Exception(path, "doesn't exist")

        if not os.path.isdir(path):
            raise Exception(path, "must be a directory")

        topology = Topology.load(path)

        model = Net()
        model.configure_topology(topology)

        return model
