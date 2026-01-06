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

from mononoqe.training.optimizers.register import register, factory

# Here a list of already implemented optimizers
# https://github.com/huggingface/pytorch-image-models/tree/main/timm/optim

ADAM_OPTIMIZER = "adam"
ADAMO_OPTIMIZER = "adamo"  # Adam 'optimized' with custom parameters
ADAMS_OPTIMIZER = "adams"  # Alias Adam + AMSGrad
ADAMW_OPTIMIZER = "adamw"  # Adam with weight regularization
ADAMSW_OPTIMIZER = "adamsw"  # Alias AdamW(eight) + AMSGrad
SGD_OPTIMIZER = "sgd"  # Vanilla Stochastic Gradient Descent
SGDM_OPTIMIZER = "sgdm"  # SGD with mementum
NAG_OPTIMIZER = "nag"  # Nesterov Accelerate Gradient


def build_optimizer(name: str, model_parameters: dict, opt_params: dict):
    opt_lambda, opt_default_params = factory()[name]()
    opt_default_params.update(opt_params)
    optimizer = opt_lambda(model_parameters, **opt_default_params)

    return optimizer


@register(ADAMO_OPTIMIZER)
def adamo_optimizer():
    # These empirically-chosen parameters will decrease historical mementum weight
    # It makes Adam more "curious" and inclined to explore a wider search space
    # Original values : (0.9, 0.999)
    return torch.optim.Adam, {
        "eps": 1e-8,
        "amsgrad": False,  # If True => Very bad results for overfitting
        "weight_decay": 0,  # If not 0 => Apply weight normalization, don't want
        "betas": (0.899, 0.99),
    }


@register(ADAM_OPTIMIZER)
def adam_optimizer():
    # https://arxiv.org/abs/1412.6980
    return torch.optim.Adam, {"eps": 1e-8, "amsgrad": False, "weight_decay": 0}


@register(ADAMS_OPTIMIZER)
def adams_optimizer():
    # http://www.satyenkale.com/papers/amsgrad.pdf
    return torch.optim.Adam, {"eps": 1e-8, "amsgrad": True, "weight_decay": 0}


@register(ADAMW_OPTIMIZER)
def adamw_optimizer():
    # https://arxiv.org/abs/1711.05101
    return torch.optim.AdamW, {"eps": 1e-8, "amsgrad": False, "weight_decay": 1e-2}


@register(ADAMSW_OPTIMIZER)
def adamsw_optimizer():
    return torch.optim.AdamW, {"eps": 1e-8, "amsgrad": True, "weight_decay": 1e-2}


@register(SGD_OPTIMIZER)
def sgd_optimizer():
    return torch.optim.SGD, {}


@register(SGDM_OPTIMIZER)
def sgdm_optimizer():
    return torch.optim.SGD, {"momentum": 0.9}


@register(NAG_OPTIMIZER)
def nesterov_accelerate_gradient_optimizer():
    return torch.optim.SGD, {"nesterov": True}
