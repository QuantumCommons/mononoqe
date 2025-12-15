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

from functools import lru_cache

from mononoqe.models.layers.quantum.post_selecters.register import register, factory


def build_post_select(name: str, nb_modes: int):
    return factory()[name](nb_modes)


@register("unique_photon_by_mode")
@lru_cache
def unique_photon_by_mode(nb_modes: int) -> str:
    post_str = [f"[{str(i)}] < 2" for i in range(nb_modes)]
    post_str = " & ".join(post_str)
    return post_str


@register("dre_even")
@lru_cache
def dre_even(nb_modes: int) -> str:
    assert nb_modes % 2 == 0

    post_str = [f"[{str(i)}] < 2" for i in range(0, nb_modes, 2)]
    post_str += [f"[{str(i)}] == 0" for i in range(1, nb_modes, 2)]
    post_str = " & ".join(post_str)

    return post_str


@register("dre_odd")
@lru_cache
def dre_odd(nb_modes: int) -> str:
    assert nb_modes % 2 == 0

    post_str = [f"[{str(i)}] == 0" for i in range(0, nb_modes, 2)]
    post_str += [f"[{str(i)}] < 2" for i in range(1, nb_modes, 2)]
    post_str = " & ".join(post_str)

    return post_str
