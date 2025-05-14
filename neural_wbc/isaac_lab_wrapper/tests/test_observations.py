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
from unittest.mock import Mock

from neural_wbc.isaac_lab_wrapper.observations import compute_privileged_observations

from isaaclab.assets import Articulation


class TestObservations(unittest.TestCase):
    def setUp(self):
        self.num_envs = 3
        num_actions = 19

        # Mocking the environment and its attributes
        # Not creating an NeuralWBCEnv instance because the delete function isn't working properly
        # due to a missing event manager in the instance, and therefore creates a conflict with
        # the one in test_neural_wbc_env.py.
        self.env = Mock()
        self.env.device = "cpu"
        self.env.contact_sensor = Mock()
        self.env.contact_sensor.data.net_forces_w = torch.randn(self.num_envs, 2, 3)
        self.env.feet_ids = [0, 1]
        self.env.base_com_bias = torch.randn((self.num_envs, 3))
        self.env.body_mass_scale = torch.randn((self.num_envs, 10))
        self.env.kp_scale = torch.randn((self.num_envs, num_actions))
        self.env.kd_scale = torch.randn((self.num_envs, num_actions))
        self.env.rfi_lim = torch.randn((self.num_envs, num_actions))
        self.env.cfg.default_rfi_lim = torch.randn((self.num_envs, num_actions))
        self.env.recovery_counters = torch.randint(0, 5, (self.num_envs,))

        # Mocking the Articulation and its attributes
        self.asset = Mock(spec=Articulation)
        self.asset.data.joint_friction = torch.randn((self.num_envs, 2))

    def test_compute_privileged_observations(self):
        privileged_obs, privileged_obs_dict = compute_privileged_observations(env=self.env, asset=self.asset)

        # Check the keys in the privileged observation dictionary
        self.assertIn("base_com_bias", privileged_obs_dict)
        self.assertIn("ground_friction_values", privileged_obs_dict)
        self.assertIn("body_mass_scale", privileged_obs_dict)
        self.assertIn("kp_scale", privileged_obs_dict)
        self.assertIn("kd_scale", privileged_obs_dict)
        self.assertIn("rfi_lim_scale", privileged_obs_dict)
        self.assertIn("contact_forces", privileged_obs_dict)
        self.assertIn("recovery_counters", privileged_obs_dict)

        # Check the tensor values
        self.assertTrue(torch.equal(privileged_obs_dict["base_com_bias"], self.env.base_com_bias))
        self.assertTrue(
            torch.equal(
                privileged_obs_dict["ground_friction_values"],
                self.asset.data.joint_friction_coeff[:, self.env.feet_ids],
            )
        )
        self.assertTrue(torch.equal(privileged_obs_dict["body_mass_scale"], self.env.body_mass_scale))
        self.assertTrue(torch.equal(privileged_obs_dict["kp_scale"], self.env.kp_scale))
        self.assertTrue(torch.equal(privileged_obs_dict["kd_scale"], self.env.kd_scale))
        self.assertTrue(
            torch.equal(privileged_obs_dict["rfi_lim_scale"], self.env.rfi_lim / self.env.cfg.default_rfi_lim)
        )
        self.assertTrue(
            torch.equal(
                privileged_obs_dict["contact_forces"],
                self.env.contact_sensor.data.net_forces_w[:, self.env.feet_ids, :].reshape(self.num_envs, -1),
            )
        )
        self.assertTrue(
            torch.equal(
                privileged_obs_dict["recovery_counters"], torch.clamp_max(self.env.recovery_counters.unsqueeze(1), 1)
            )
        )

        # Check the concatenated privileged observations tensor
        expected_privileged_obs = torch.cat(
            [
                privileged_obs_dict["base_com_bias"],
                privileged_obs_dict["ground_friction_values"],
                privileged_obs_dict["body_mass_scale"],
                privileged_obs_dict["kp_scale"],
                privileged_obs_dict["kd_scale"],
                privileged_obs_dict["rfi_lim_scale"],
                privileged_obs_dict["contact_forces"],
                privileged_obs_dict["recovery_counters"],
            ],
            dim=-1,
        )
        self.assertTrue(torch.equal(privileged_obs, expected_privileged_obs))
