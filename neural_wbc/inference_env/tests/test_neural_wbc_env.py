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

from inference_env.neural_wbc_env import NeuralWBCEnv
from inference_env.neural_wbc_env_cfg_h1 import NeuralWBCEnvCfgH1

from neural_wbc.data import get_data_path


class TestNeuralWBCEnv(unittest.TestCase):
    def setUp(self):
        # create environment configuration
        self.env_cfg = NeuralWBCEnvCfgH1(model_xml_path=get_data_path("mujoco/models/h1.xml"))

        # setup RL environment
        self.device = torch.device("cpu")
        self.env = NeuralWBCEnv(cfg=self.env_cfg, device=self.device)

        self.num_instances = self.env.robot.num_instances
        self.num_controls = self.env.robot.num_controls

    def test_reset(self):
        # Reset without specifying reset positions
        joint_pos_before_reset = self.env.robot.joint_positions.clone()
        self.env.reset()
        joint_pos_after_reset = self.env.robot.joint_positions.clone()
        self.assertTrue(torch.equal(joint_pos_before_reset, joint_pos_after_reset))

        # Reset WBCMujoco with specified reset positions
        joint_pos_before_reset = self.env.robot.joint_positions.clone()
        reset_pos = (
            torch.from_numpy(self.env.robot.internal_sim.data.qpos.copy())
            .to(dtype=torch.float, device=self.device)
            .expand(1, -1)
        )
        reset_pos[:, : self.num_controls] = torch.rand(self.num_instances, self.num_controls, device=self.device)
        self.env.robot.reset(qpos=reset_pos)
        joint_pos_after_reset = self.env.robot.joint_positions.clone()
        self.assertFalse(torch.equal(joint_pos_before_reset, joint_pos_after_reset))

    def test_step(self):
        joint_pos_before_step = self.env.robot.joint_positions.clone()
        actions = torch.zeros(self.num_instances, self.num_controls, device=self.device)
        self.env.step(actions=actions)
        joint_pos_after_step = self.env.robot.joint_positions.clone()
        self.assertFalse(torch.equal(joint_pos_before_step, joint_pos_after_step))

    def test_action_delay(self):
        # Set the action delay.
        self.env_cfg.ctrl_delay_step_range = [1, 1]
        self.env = NeuralWBCEnv(cfg=self.env_cfg, device=self.device)
        actions = torch.rand(self.num_instances, self.num_controls, device=self.device)
        self.env.step(actions=actions)
        # First action should be zero.
        self.assertTrue(torch.equal(self.env.actions, torch.zeros_like(actions)))

        # Now should be equal.
        self.env.step(actions=actions)
        self.assertTrue(torch.equal(self.env.actions, actions))

    def test_effort_limit(self):
        # Disable the action delay to test action limit.
        self.env_cfg.ctrl_delay_step_range = [0, 0]
        self.env = NeuralWBCEnv(cfg=self.env_cfg, device=self.device)
        # Create a very large value action and make sure it is properly clamped
        actions = 1000 * torch.ones(self.num_instances, self.num_controls, device=self.device)
        self.env.step(actions=actions)

        effort_limits = torch.zeros_like(actions)
        for joint_name, effort_limit in self.env_cfg.effort_limit.items():
            joint_id = self.env.robot.get_joint_ids([joint_name])[joint_name]
            effort_limits[:, joint_id] = effort_limit

        self.assertTrue(torch.equal(self.env._processed_action, effort_limits))


if __name__ == "__main__":
    unittest.main()
