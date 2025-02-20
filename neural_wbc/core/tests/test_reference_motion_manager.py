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
import unittest

from neural_wbc.core.reference_motion import ReferenceMotionManager, ReferenceMotionManagerCfg, ReferenceMotionState
from neural_wbc.data import get_data_path


class TestReferenceMotionManager(unittest.TestCase):
    num_envs = 10
    device = torch.device("cpu")

    def _create_reference_motion_manager(self):
        cfg = ReferenceMotionManagerCfg()
        cfg.motion_path = get_data_path("motions/stable_punch.pkl")
        cfg.skeleton_path = get_data_path("motion_lib/h1.xml")

        return ReferenceMotionManager(
            cfg=cfg,
            device=self.device,
            num_envs=self.num_envs,
            random_sample=False,
            extend_head=False,
            dt=0.005,
            # TODO: add RayCaster
        )

    def test_load_reference_motion(self):
        mgr = self._create_reference_motion_manager()
        self.assertEqual(mgr.motion_lib.motion_ids.numel(), self.num_envs)

    def test_get_state_from_motion_lib_cache(self):
        mgr = self._create_reference_motion_manager()
        env_ids = torch.arange(self.num_envs).long()
        mgr.reset_motion_start_times(env_ids=env_ids, sample=True)
        episode_length_buf = torch.Tensor([0] * self.num_envs, device=self.device).long()
        ref_motion_state = mgr.get_state_from_motion_lib_cache(episode_length_buf=episode_length_buf)
        for attr_name in dir(ref_motion_state):
            if attr_name.startswith("__"):
                continue
            value = getattr(ref_motion_state, attr_name)
            self.assertEqual(value.shape[0], self.num_envs)

    def test_reference_motion_state(self):
        mgr = self._create_reference_motion_manager()
        env_ids = torch.arange(self.num_envs).long()
        mgr.reset_motion_start_times(env_ids=env_ids, sample=False)
        episode_length_buf = torch.Tensor([0] * self.num_envs, device=self.device).long()
        ref_motion_state: ReferenceMotionState = mgr.get_state_from_motion_lib_cache(
            episode_length_buf=episode_length_buf
        )
        self.assertAlmostEqual(ref_motion_state.body_rot[0, 0, 0], 1)
        self.assertAlmostEqual(ref_motion_state.body_rot[0, 0, 1], 0)
        self.assertAlmostEqual(ref_motion_state.body_rot[0, 0, 2], 0)
        self.assertAlmostEqual(ref_motion_state.body_rot[0, 0, 3], 0)

    def test_reset_motion_start_times_after_load_motion(self):
        mgr = self._create_reference_motion_manager()
        env_ids = torch.arange(self.num_envs).long()
        mgr.reset_motion_start_times(env_ids=env_ids, sample=True)
        mgr.load_motions(random_sample=False, start_idx=0)
        self.assertTrue(torch.all(mgr.motion_start_times == 0))

    def test_reference_motion_state_reset(self):
        mgr = self._create_reference_motion_manager()
        episode_length_buf = torch.Tensor([0] * self.num_envs, device=self.device).long()
        # Read the first frame
        ref_motion_state1: ReferenceMotionState = mgr.get_state_from_motion_lib_cache(
            episode_length_buf=episode_length_buf
        )
        # Load a new motion and read the first frame.
        mgr.load_motions(random_sample=False, start_idx=1)
        ref_motion_state2: ReferenceMotionState = mgr.get_state_from_motion_lib_cache(
            episode_length_buf=episode_length_buf
        )
        # Confirm that the motion is updated properly
        self.assertFalse(torch.equal(ref_motion_state1.joint_pos, ref_motion_state2.joint_pos))
