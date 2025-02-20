# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

import mujoco as mj
import mujoco_viewer.mujoco_viewer as mjv

from neural_wbc.core.reference_motion import ReferenceMotionState


class MujocoVisualizer:
    """A basic Mujoco visualizer"""

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        width: int = 1400,
        height: int = 1200,
        start_paused: bool = True,
    ):
        self._model = model
        self._data = data
        self._viewer = mjv.MujocoViewer(
            self._model,
            self._data,
            width=width,
            height=height,
            hide_menus=True,
        )

        self._viewer._paused = start_paused  # pylint: disable=W0212

    def update(self):
        """Update the Mujoco viewer."""
        if self._viewer.is_alive:
            self._viewer.render()

    def draw_reference_state(self, state: ReferenceMotionState):
        """Visualize the reference state in Mujoco."""

        body_pos_np = np.squeeze(state.body_pos.detach().cpu().numpy())
        body_pos_extend_np = np.squeeze(state.body_pos_extend.detach().cpu().numpy())
        body_pos = np.vstack([body_pos_np, body_pos_extend_np])

        for i in range(body_pos.shape[0]):
            self._viewer.add_marker(
                pos=body_pos[i],
                size=0.05,
                rgba=(1, 0, 0, 1),
                type=mj.mjtGeom.mjGEOM_SPHERE,
                label="",
                id=i,
            )

    def close(self):
        self._viewer.close()
