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

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
import shutil

from mononoqe.data import TrainingData
from mononoqe.models import Net
from mononoqe.models.topologies import TopologyParams
from mononoqe.training import TrainingParams


_TENSORBOARD_DIRECTORY = "tb_logs"


class _TrainingCallback(Callback):
    def on_train_epoch_end(self, trainer: pl.Trainer, model: Net):
        print(
            f"epoch {trainer.current_epoch}: val_accuracy={100*trainer.callback_metrics['val_accuracy']:.0f}%"
        )


class Trainer:
    def __init__(self) -> "Trainer":
        pass

    def fit(
        self,
        model: Net,
        topology_params: TopologyParams,
        training_params: TrainingParams,
        training_data: TrainingData,
        device: str,
    ):
        shutil.rmtree(_TENSORBOARD_DIRECTORY, ignore_errors=True)

        # Configure main training loop system:
        #   Put custom callback if necessary
        #   Add logging system, can be visualized with `make board` then go to localhost:6006
        accelerator = "cpu" if device == "cpu" else "auto"
        trainer = pl.Trainer(
            max_epochs=training_params.epochs,
            logger=TensorBoardLogger(save_dir=_TENSORBOARD_DIRECTORY, name="qml-mnist"),
            callbacks=[_TrainingCallback()],
            accelerator=accelerator,
        )

        # Get all data for training
        train_dataloader, validation_dataloader, data_input_shape, data_output_shape = (
            training_data.build_loaders()
        )

        # Configure the model topology accordingly to expected input and output shapes
        model.configure_topology(
            TopologyParams(
                name=topology_params.name,
                extra=topology_params.extra,
                input_shape=data_input_shape,
                output_shape=data_output_shape,
            )
        )

        # Configure training hyperparameteers (loss, optimizer...) based on model parameters (weigths)
        model.configure_training(training_params)

        # Engage the model training based on data
        trainer.fit(model, train_dataloader, validation_dataloader)
