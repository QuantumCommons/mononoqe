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
import torch.optim.lr_scheduler as lr_scheduler

from mononoqe.utils import Factory


__FACTORY = Factory("scheduler")

def factory() -> Factory:
    global __FACTORY
    return __FACTORY

def register(name: str):
    return factory().register(name)


# Here a list of already implemented scheduler:
# https://docs.pytorch.org/docs/2.9/optim.html#how-to-adjust-learning-rate

STEPLR_SCHEDULER = "steplr"
POLYLR_SCHEDULER = "polylr"


def build_scheduler(name: str, optimizer: torch.optim.Optimizer, **kwargs) -> lr_scheduler.LRScheduler:
    scheduler: lr_scheduler.LRScheduler = factory()[name](optimizer, **kwargs)
    return scheduler


@register(STEPLR_SCHEDULER)
def step_scheduler(optimizer, **kwargs):
    return lr_scheduler.StepLR(optimizer, **kwargs)


@register(POLYLR_SCHEDULER)
def poly_scheduler(optimizer, **kwargs):
    return lr_scheduler.PolynomialLR(optimizer, **kwargs)
