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
from unittest.mock import MagicMock, mock_open, patch

from ..teacher_policy_cfg import AlgorithmCfg, PolicyCfg, RunnerCfg, TeacherPolicyCfg
from ..utils import get_customized_rsl_rl, get_ppo_runner_and_checkpoint_path

get_customized_rsl_rl()
from rsl_rl.runners import OnPolicyRunner


class TestLoadFromCheckpoint(unittest.TestCase):
    # Mocking open and load functions to avoid specifying files
    @patch("builtins.open", new_callable=mock_open, read_data='{"policy": {}}')
    @patch("json.load")
    @patch("torch.optim.Adam")
    def test_load_from_checkpoint_success(self, mock_adam, mock_json_load, mock_open):
        # Mock arguments
        teacher_policy_cfg = TeacherPolicyCfg(
            seed=1,
            policy=PolicyCfg(),
            algorithm=AlgorithmCfg(gamma=1.9),
            runner=RunnerCfg(resume_path="/path/to/resume", checkpoint="checkpoint_file", num_steps_per_env=38),
        )
        wrapped_env = MagicMock()
        wrapped_env.reset = MagicMock(return_value=[None, None])
        device = torch.device("cpu")
        log_dir = "/path/to/log"

        # Mock return values
        mock_json_load.return_value = {"policy": {"init_noise_std": 10.0, "activation": "sigmoid"}}

        # Call the function
        ppo_runner, checkpoint_path = get_ppo_runner_and_checkpoint_path(
            teacher_policy_cfg, wrapped_env, device, log_dir
        )

        # Assertions
        self.assertEqual(checkpoint_path, "/path/to/resume/checkpoint_file")
        self.assertIsInstance(ppo_runner, OnPolicyRunner)
        mock_open.assert_called_once_with("/path/to/resume/config.json", encoding="utf-8")

        # Check overwrite values
        self.assertEqual(teacher_policy_cfg.policy.init_noise_std, 10.0)
        self.assertEqual(teacher_policy_cfg.policy.activation, "sigmoid")

        # Compare teacher policy config and OnPolicyPlayer's config
        teacher_policy_dict = teacher_policy_cfg.to_dict()
        self.assertDictEqual(teacher_policy_dict["policy"], ppo_runner.policy_cfg)
        self.assertDictEqual(teacher_policy_dict["algorithm"], ppo_runner.alg_cfg)
        self.assertDictEqual(teacher_policy_dict["runner"], ppo_runner.cfg)

    def test_load_from_checkpoint_no_resume_path(self):
        teacher_policy_cfg = TeacherPolicyCfg(
            seed=1,
            policy=PolicyCfg(),
            algorithm=AlgorithmCfg(),
            runner=RunnerCfg(resume_path="", checkpoint="checkpoint_file"),
        )
        wrapped_env = MagicMock()
        device = torch.device("cpu")
        log_dir = "/path/to/log"

        with self.assertRaises(ValueError) as context:
            get_ppo_runner_and_checkpoint_path(teacher_policy_cfg, wrapped_env, device, log_dir)
        self.assertEqual(str(context.exception), "teacher_policy.resume_path is not specified")

    def test_load_from_checkpoint_no_checkpoint(self):
        teacher_policy_cfg = TeacherPolicyCfg(
            seed=1,
            policy=PolicyCfg(),
            algorithm=AlgorithmCfg(),
            runner=RunnerCfg(resume_path="/path/to/resume", checkpoint=""),
        )
        wrapped_env = MagicMock()
        device = torch.device("cpu")
        log_dir = "/path/to/log"

        with self.assertRaises(ValueError) as context:
            get_ppo_runner_and_checkpoint_path(teacher_policy_cfg, wrapped_env, device, log_dir)
        self.assertEqual(str(context.exception), "teacher_policy.checkpoint is not specified")
