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


import json
import os
import shutil
import torch
import unittest

from neural_wbc.student_policy.student_policy_trainer import StudentPolicyTrainer, StudentPolicyTrainerCfg
from neural_wbc.student_policy.teacher_policy import TeacherPolicy


class MockedEnvCfg:
    """
    A minimal configuration that includes only the essential fields required for unit-testing the StudentPolicyTrainer class.
    """

    distill = True


class MockedEnv:
    """
    A minimal environment designed to include only the essential fields needed for unit-testing the StudentPolicyTrainer class.
    """

    def __init__(self):
        self.num_envs = 23
        self.num_obs = 913
        self.num_privileged_obs = 990
        self.num_student_obs = 1665
        self.num_actions = 19
        self.device = torch.device("cpu")

        self.cfg = MockedEnvCfg()

        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device)
        self.student_obs_buf = torch.zeros(self.num_envs, self.num_student_obs, device=self.device)
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, 1, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, 1, device=self.device)
        self.extras = {"observations": self._generate_random_observations()}

        # Required by rsl_rl
        self.kin_dict = {"gt_action": torch.zeros([self.num_envs, self.num_actions]).to(self.device)}

    def _generate_random_observations(self):
        obs = {
            "teacher_policy": torch.randn(self.num_envs, self.num_obs, device=self.device),
            "critic": torch.randn(self.num_envs, self.num_privileged_obs, device=self.device),
            "student_policy": torch.randn(self.num_envs, self.num_student_obs, device=self.device),
        }
        self.obs_buf = obs["teacher_policy"]
        self.privileged_obs_buf = obs["critic"]
        self.student_obs_buf = obs["student_policy"]
        return obs

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs = self._generate_random_observations()

        self.rew_buf = torch.randn(self.num_envs, device=self.device)
        self.reset_buf = torch.randn(self.num_envs) < 0.5
        self.extras["observations"] = obs

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        return self.obs_buf, self.privileged_obs_buf

    def get_full_observations(self) -> dict[str, torch.Tensor]:
        return self._generate_random_observations()

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self) -> torch.Tensor:
        return self.privileged_obs_buf


class MockedExpertPolicy(TeacherPolicy):
    def __init__(self, num_envs, num_actions, device):
        super().__init__()
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.device = device

    def load(self, path, **kwargs):
        return None

    def act_rollout(self, observations, **kwargs):
        return torch.zeros(self.num_envs, self.num_actions, device=self.device)

    def act(self, observations, **kwargs):
        return torch.zeros(self.num_envs, self.num_actions, device=self.device)


class TestStudentPolicyTrainer(unittest.TestCase):
    """Unit tests for the StudentPolicyTrainer class."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.student_policy_path = "/tmp/student_policy_unit_tests"
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.student_policy_path, ignore_errors=True)
        return super().tearDownClass()

    def test_run(self):
        env = MockedEnv()
        teacher_policy = MockedExpertPolicy(env.num_envs, env.num_actions, env.device)
        cfg = StudentPolicyTrainerCfg(
            teacher_policy=teacher_policy,
            student_policy_path=self.student_policy_path,
            num_steps_per_env=24,
            num_policy_obs=env.num_obs,
            num_student_obs=env.num_student_obs,
            num_actions=env.num_actions,
            max_iteration=2,
        )
        runner = StudentPolicyTrainer(env=env, cfg=cfg)
        runner.run()

        self.assertTrue(os.path.exists(self.student_policy_path))
        config_path = os.path.join(self.student_policy_path, "config.json")
        with open(config_path) as fh:
            config_dict = json.load(fh)
        config_dict["teacher_policy"] = teacher_policy
        reconstructed_cfg = StudentPolicyTrainerCfg(**config_dict)
        self.assertEqual(cfg, reconstructed_cfg)

    def test_resume_path_and_checkpoint(self):
        env = MockedEnv()
        teacher_policy = MockedExpertPolicy(env.num_envs, env.num_actions, env.device)

        # Setting resume_path in the config but not the checkpoint results in a ValueError
        cfg = StudentPolicyTrainerCfg(
            teacher_policy=teacher_policy,
            student_policy_path=self.student_policy_path,
            num_steps_per_env=24,
            num_policy_obs=env.num_obs,
            num_student_obs=env.num_student_obs,
            num_actions=env.num_actions,
            max_iteration=2,
            resume_path="fake/path",
        )
        self.assertRaises(ValueError, StudentPolicyTrainer, env=env, cfg=cfg)

        # Setting checkpoint without a resume_path results in a ValueError
        cfg.resume_path = ""
        cfg.checkpoint = "fake_model.pt"
        self.assertRaises(ValueError, StudentPolicyTrainer, env=env, cfg=cfg)

        # Setting fake resume path and/or fake checkpoint results in FileNotFoundError
        cfg.resume_path = "fake/path"
        self.assertRaises(FileNotFoundError, StudentPolicyTrainer, env=env, cfg=cfg)


if __name__ == "__main__":
    unittest.main()
