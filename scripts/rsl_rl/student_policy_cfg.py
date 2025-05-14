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

from dataclasses import MISSING

from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg, RslRlDistillationStudentTeacherCfg, RslRlOnPolicyRunnerCfg

from isaaclab.utils import configclass


@configclass
class RslDistillationAlgorithm_Extended_Cfg(RslRlDistillationAlgorithmCfg):
    """Configuration settings for the distillation algorithm."""

    # Add max_grad_norm to the configuration
    # As it has not been added to isaaclab yet
    max_grad_norm: float = MISSING
    pass


@configclass
class StudentPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration settings for the runner."""

    num_steps_per_env = 1
    max_iterations = 200000
    save_interval = 500
    experiment_name = "student_policy"
    empirical_normalization = False
    policy = RslRlDistillationStudentTeacherCfg(
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=0.001,
    )
    algorithm = RslDistillationAlgorithm_Extended_Cfg(
        num_learning_epochs=5,
        learning_rate=5e-04,
        gradient_length=1,
        max_grad_norm=0.2,
    )
