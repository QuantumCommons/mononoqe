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
import perceval as pcvl
import perceval.providers.scaleway as scw
from perceval.providers.scaleway.scaleway_rpc_handler import _DEFAULT_URL


def build_session(name: str, platform: str) -> pcvl.ISession:
    session = scw.Session(
        deduplication_id=name,
        platform=platform,
        project_id=os.environ["SCW_PROJECT_ID"],
        token=os.environ["SCW_SECRET_KEY"],
        url=os.environ.get("SCW_API_GATEWAY", _DEFAULT_URL),
    )

    return session
