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
import argparse
import tempfile
import unittest

from ..teacher_policy_cfg import AlgorithmCfg, PolicyCfg, RunnerCfg, TeacherPolicyCfg


class TestTeacherPolicyCfg(unittest.TestCase):
    def setUp(self):
        """Set up a default configuration for testing."""
        self.config = TeacherPolicyCfg(runner=RunnerCfg(path="/tmp"))

    def test_save_and_load(self):
        """Test saving and loading the configuration to/from a file."""
        # Create a temporary file to save the configuration
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            file_path = tmp_file.name

        # Save the configuration to the file
        self.config.save(file_path)

        # Load the configuration from the file
        loaded_config = TeacherPolicyCfg.load(file_path)

        # Compare the original and loaded configurations
        self.assertEqual(self.config, loaded_config)

    def test_add_args_to_parser(self):
        """Test adding configuration fields to an argument parser."""
        parser = argparse.ArgumentParser(description="RSL-RL configuration")
        TeacherPolicyCfg.add_args_to_parser(parser)

        # Create sample arguments
        args = [
            "--teacher_policy.seed",
            "42",
            "--teacher_policy.init_noise_std",
            "2.0",
            "--teacher_policy.actor_hidden_dims",
            "256",
            "128",
            "--teacher_policy.learning_rate",
            "0.0005",
            "--teacher_policy.max_iterations",
            "5000",
            "--teacher_policy.path",
            "/tmp",
        ]

        # Parse the arguments
        parsed_args = parser.parse_args(args)

        # Verify the parsed arguments using getattr
        self.assertEqual(getattr(parsed_args, "teacher_policy.seed"), 42)
        self.assertEqual(getattr(parsed_args, "teacher_policy.init_noise_std"), 2.0)
        self.assertEqual(getattr(parsed_args, "teacher_policy.actor_hidden_dims"), [256, 128])
        self.assertEqual(getattr(parsed_args, "teacher_policy.learning_rate"), 0.0005)
        self.assertEqual(getattr(parsed_args, "teacher_policy.max_iterations"), 5000)

    def test_default_values(self):
        """Test the default values of the configuration."""
        # Verify default values
        self.assertEqual(self.config.seed, 1)
        self.assertEqual(self.config.policy.init_noise_std, 1.0)
        self.assertEqual(self.config.policy.actor_hidden_dims, [512, 256, 128])
        self.assertEqual(self.config.algorithm.learning_rate, 1.0e-3)
        self.assertEqual(self.config.runner.max_iterations, 10000000)

    def test_add_args_to_parser_with_default_overwrites(self):
        """Test adding configuration fields to an argument parser with default overwrites."""
        parser = argparse.ArgumentParser(description="RSL-RL configuration")
        default_overwrites = {
            "seed": 10,
            "init_noise_std": 0.5,
            "learning_rate": 0.0015,
            "max_iterations": 2000,
        }
        TeacherPolicyCfg.add_args_to_parser(parser, default_overwrites)

        # Parse the arguments with no optional arguments
        parsed_args = parser.parse_args(
            [
                "--teacher_policy.path",
                "/tmp",
            ]
        )

        # Verify the overwritten default values
        self.assertEqual(getattr(parsed_args, "teacher_policy.seed"), default_overwrites["seed"])
        self.assertEqual(getattr(parsed_args, "teacher_policy.init_noise_std"), default_overwrites["init_noise_std"])
        self.assertEqual(
            getattr(parsed_args, "teacher_policy.learning_rate"),
            default_overwrites["learning_rate"],
        )
        self.assertEqual(getattr(parsed_args, "teacher_policy.max_iterations"), default_overwrites["max_iterations"])

    def test_policy_activation(self):
        for activation_type in PolicyCfg.ACTIVATION_TYPES:
            cfg = PolicyCfg(activation=activation_type)
            self.assertEqual(cfg.activation, activation_type)

        with self.assertRaises(ValueError):
            cfg = PolicyCfg(activation="invalid_type")

    def test_from_argparse_args_default(self):
        parser = argparse.ArgumentParser(description="Teacher Policy Configuration")
        TeacherPolicyCfg.add_args_to_parser(parser)
        args = parser.parse_args(
            [
                "--teacher_policy.path",
                "/tmp",
            ]
        )
        cfg = TeacherPolicyCfg.from_argparse_args(args)

        # Create expected default configuration
        expected_cfg = TeacherPolicyCfg(
            seed=1, policy=PolicyCfg(), algorithm=AlgorithmCfg(), runner=RunnerCfg(path="/tmp")
        )

        self.assertEqual(cfg, expected_cfg)

    def test_from_argparse_args_custom(self):
        parser = argparse.ArgumentParser(description="Teacher Policy Configuration")
        TeacherPolicyCfg.add_args_to_parser(parser)
        test_args = [
            "--teacher_policy.seed",
            "42",
            "--teacher_policy.init_noise_std",
            "0.5",
            "--teacher_policy.activation",
            "relu",
            "--teacher_policy.value_loss_coef",
            "0.5",
            "--teacher_policy.max_iterations",
            "500",
            "--teacher_policy.path",
            "/tmp",
        ]
        args = parser.parse_args(test_args)
        cfg = TeacherPolicyCfg.from_argparse_args(args)

        # Create expected custom configuration
        expected_cfg = TeacherPolicyCfg(
            seed=42,
            policy=PolicyCfg(init_noise_std=0.5, activation="relu"),
            algorithm=AlgorithmCfg(value_loss_coef=0.5),
            runner=RunnerCfg(path="/tmp", max_iterations=500),
        )

        self.assertEqual(cfg, expected_cfg)


if __name__ == "__main__":
    unittest.main()
