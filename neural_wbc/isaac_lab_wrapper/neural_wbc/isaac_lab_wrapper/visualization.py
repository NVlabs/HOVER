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

import torch

from neural_wbc.core import ReferenceMotionState

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import DEFORMABLE_TARGET_MARKER_CFG


class RefMotionVisualizer:
    """Visualizer of the reference motions."""

    def __init__(self):
        self._initialized = False
        self._active_ref_motion_markers = None
        self._inactive_ref_motion_markers = None

    def _initialize_ref_motion_markers(self):
        print("Initialize markers for ref motion joints.")
        # Visualizer for the active reference body positions.
        active_marker_cfg = DEFORMABLE_TARGET_MARKER_CFG.copy()
        active_marker_cfg.markers["target"].radius = 0.05
        active_marker_cfg.markers["target"].visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.0, 1.0)
        )  # blue
        active_marker_cfg.prim_path = "/Visuals/Command/active_ref_motion"
        self._active_ref_motion_markers = VisualizationMarkers(active_marker_cfg)
        self._active_ref_motion_markers.set_visibility(True)

        # Visualizere for the inactive reference body positions.
        inactive_marker_cfg = DEFORMABLE_TARGET_MARKER_CFG.copy()
        inactive_marker_cfg.markers["target"].radius = 0.03
        inactive_marker_cfg.markers["target"].visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0)
        )  # red
        inactive_marker_cfg.prim_path = "/Visuals/Command/inactive_ref_motion"
        self._inactive_ref_motion_markers = VisualizationMarkers(inactive_marker_cfg)
        self._inactive_ref_motion_markers.set_visibility(True)

        self._initialized = True

    def visualize(self, ref_motion: ReferenceMotionState, mask: torch.Tensor | None = None):
        if not self._initialized:
            self._initialize_ref_motion_markers()

        # Split the active and inactive body references.
        if mask is None:
            active_body_pos = ref_motion.body_pos_extend.view(-1, 3)
            device = active_body_pos.device
            inactive_body_pos = torch.zeros(0, 3, device=device)
        else:
            num_bodies = ref_motion.body_pos_extend.shape[1]
            mask = mask[:, :num_bodies]
            # We reshape to still have the correct shape even if the mask is set to all false.
            # Else we will try to visualize non-existing bodies below.
            mask_flat = mask.flatten()
            active_body_pos = ref_motion.body_pos_extend.view(-1, 3)[mask_flat, :]
            inactive_body_pos = ref_motion.body_pos_extend.view(-1, 3)[~mask_flat, :]

        # Update the position of the visualization markers.
        if active_body_pos.shape[0] != 0:
            # Need to set visibility to True to ensure the markers are visible.
            self._active_ref_motion_markers.set_visibility(True)
            self._active_ref_motion_markers.visualize(active_body_pos)
        else:
            self._active_ref_motion_markers.set_visibility(False)
        if inactive_body_pos.shape[0] != 0:
            # Need to set visibility to True to ensure the markers are visible.
            self._inactive_ref_motion_markers.set_visibility(True)
            self._inactive_ref_motion_markers.visualize(inactive_body_pos)
        else:
            self._inactive_ref_motion_markers.set_visibility(False)
