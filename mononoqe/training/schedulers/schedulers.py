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

import timm.scheduler as sch
import torch

from mononoqe.training.schedulers.register import register, factory

# Here a list of already implemented scheduler:
# https://github.com/huggingface/pytorch-image-models/tree/main/timm/scheduler

TANH_SCHEDULER = "tanh"
POLYLR_SCHEDULER = "polylr"


def build_scheduler(name: str, optimizer):
    scheduler = factory()[name](optimizer)
    return scheduler


@register(TANH_SCHEDULER)
def tanh_scheduler(optimizer):
    return sch.TanhLRScheduler(optimizer, t_initial=1)


@register(POLYLR_SCHEDULER)
def tanh_scheduler(optimizer):
    return sch.PolyLRScheduler(optimizer, t_initial=1)
