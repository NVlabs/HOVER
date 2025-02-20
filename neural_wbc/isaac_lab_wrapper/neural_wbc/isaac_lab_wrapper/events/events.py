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


"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING, Literal

from neural_wbc.core import ReferenceMotionState, math_utils

import isaaclab.envs.mdp as mdp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from ..neural_env import NeuralWBCEnv


def randomize_body_com(
    env: NeuralWBCEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    distribution_params: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the com of the bodies by adding, scaling or setting random values.

    This function allows randomizing the center of mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales or sets the values into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current masses of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms()

    if not hasattr(env, "default_coms"):
        # Randomize robot base com
        env.default_coms = coms.clone()
        env.base_com_bias = torch.zeros((env.num_envs, 3), dtype=torch.float, device=coms.device)

    # apply randomization on default values
    coms[env_ids[:, None], body_ids] = env.default_coms[env_ids[:, None], body_ids].clone()

    dist_fn = resolve_dist_fn(distribution)

    if isinstance(distribution_params[0], torch.Tensor):
        distribution_params = (distribution_params[0].to(coms.device), distribution_params[1].to(coms.device))

    env.base_com_bias[env_ids, :] = dist_fn(
        *distribution_params, (env_ids.shape[0], env.base_com_bias.shape[1]), device=coms.device
    )

    # sample from the given range
    if operation == "add":
        coms[env_ids[:, None], body_ids, :3] += env.base_com_bias[env_ids[:, None], :]
    elif operation == "abs":
        coms[env_ids[:, None], body_ids, :3] = env.base_com_bias[env_ids[:, None], :]
    elif operation == "scale":
        coms[env_ids[:, None], body_ids, :3] *= env.base_com_bias[env_ids[:, None], :]
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_pd_scale(
    env: NeuralWBCEnv,
    env_ids: torch.Tensor | None,
    distribution_params: tuple[float, float],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the scale of the pd gains by adding, scaling or setting random values.

    This function allows randomizing the scale of the pd gain. The function samples random values from the
    given distribution parameters and adds, or sets the values into the simulation based on the operation.

    """
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # apply randomization on default values
    kp_scale = env.default_kp_scale.clone()
    kd_scale = env.default_kd_scale.clone()

    dist_fn = resolve_dist_fn(distribution)

    # sample from the given range
    if operation == "add":
        kp_scale[env_ids, :] += dist_fn(
            *distribution_params, (env_ids.shape[0], kp_scale.shape[1]), device=kp_scale.device
        )
        kd_scale[env_ids, :] += dist_fn(
            *distribution_params, (env_ids.shape[0], kd_scale.shape[1]), device=kd_scale.device
        )
    elif operation == "abs":
        kp_scale[env_ids, :] = dist_fn(
            *distribution_params, (env_ids.shape[0], kp_scale.shape[1]), device=kp_scale.device
        )
        kd_scale[env_ids, :] = dist_fn(
            *distribution_params, (env_ids.shape[0], kd_scale.shape[1]), device=kd_scale.device
        )
    elif operation == "scale":
        kp_scale[env_ids, :] *= dist_fn(
            *distribution_params, (env_ids.shape[0], kp_scale.shape[1]), device=kp_scale.device
        )
        kd_scale[env_ids, :] *= dist_fn(
            *distribution_params, (env_ids.shape[0], kd_scale.shape[1]), device=kd_scale.device
        )
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    env.kp_scale[env_ids, :] = kp_scale[env_ids, :]
    env.kd_scale[env_ids, :] = kd_scale[env_ids, :]


def randomize_action_noise_range(
    env: NeuralWBCEnv,
    env_ids: torch.Tensor | None,
    distribution_params: tuple[float, float],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the sample range of the added action noise by adding, scaling or setting random values.

    This function allows randomizing the scale of the sample range of the added action noise. The function
    samples random values from the given distribution parameters and adds, scales or sets the values into the
    simulation based on the operation.

    """

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # apply randomization on default values
    rfi_lim = env.default_rfi_lim.clone()

    dist_fn = resolve_dist_fn(distribution)

    # sample from the given range
    if operation == "add":
        rfi_lim[env_ids, :] += dist_fn(
            *distribution_params, (env_ids.shape[0], rfi_lim.shape[1]), device=rfi_lim.device
        )
    elif operation == "abs":
        rfi_lim[env_ids, :] = dist_fn(*distribution_params, (env_ids.shape[0], rfi_lim.shape[1]), device=rfi_lim.device)
    elif operation == "scale":
        rfi_lim[env_ids, :] *= dist_fn(
            *distribution_params, (env_ids.shape[0], rfi_lim.shape[1]), device=rfi_lim.device
        )
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    env.rfi_lim[env_ids, :] = rfi_lim[env_ids, :]


def randomize_motion_ref_xyz(
    env: NeuralWBCEnv,
    env_ids: torch.Tensor | None,
    distribution_params: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the motion reference x,y,z offset by adding, scaling or setting random values.

    This function allows randomizing the motion reference x,y,z offset. The function samples
    random values from the given distribution parameters and adds, scales or sets the values into the
    simulation based on the operation.

    """

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # apply randomization on default values
    ref_episodic_offset = env.default_ref_episodic_offset.clone()

    dist_fn = resolve_dist_fn(distribution)

    if isinstance(distribution_params[0], torch.Tensor):
        distribution_params = (
            distribution_params[0].to(ref_episodic_offset.device),
            distribution_params[1].to(ref_episodic_offset.device),
        )

    # sample from the given range
    if operation == "add":
        ref_episodic_offset[env_ids, :] += dist_fn(
            *distribution_params,
            (env_ids.shape[0], ref_episodic_offset.shape[1]),
            device=ref_episodic_offset.device,
        )
    elif operation == "abs":
        ref_episodic_offset[env_ids, :] = dist_fn(
            *distribution_params,
            (env_ids.shape[0], ref_episodic_offset.shape[1]),
            device=ref_episodic_offset.device,
        )
    elif operation == "scale":
        ref_episodic_offset[env_ids, :] *= dist_fn(
            *distribution_params,
            (env_ids.shape[0], ref_episodic_offset.shape[1]),
            device=ref_episodic_offset.device,
        )
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    env.ref_episodic_offset[env_ids, :] = ref_episodic_offset[env_ids, :]


def push_by_setting_velocity_with_recovery(
    env: NeuralWBCEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_w[:] = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
    # give pushed robot time to recover
    env.recovery_counters[env_ids] = env.cfg.recovery_count


def reset_robot_state_and_motion(
    env: NeuralWBCEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot and reference motion.

    The full reset percess is:
    1. Reset the robot root state to the origin position of its terrain.
    2. Reset motion reference to random time step.
    3. Moving the motion reference trajectory to the current robot position.
    4. Reset the robot joint to reference motion's joint states
    5. Reset the robot root state to the reference motion's root state.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()
    root_states[:, :3] += env._terrain.env_origins[env_ids]
    state = root_states[:, :7]
    asset.write_root_pose_to_sim(state, env_ids=env_ids)

    mdp.reset_joints_by_scale(env, env_ids, (1.0, 1.0), (0.0, 0.0), asset_cfg)

    # Sample new commands
    env._ref_motion_mgr.reset_motion_start_times(env_ids=env_ids, sample=env.cfg.mode.is_training_mode())

    # Record new start locations on terrain
    env._start_positions_on_terrain[env_ids, ...] = root_states[:, :3]

    ref_motion_state: ReferenceMotionState = env._ref_motion_mgr.get_state_from_motion_lib_cache(
        episode_length_buf=0,
        offset=env._start_positions_on_terrain,
        terrain_heights=env.get_terrain_heights(),
    )

    env._ref_motion_visualizer.visualize(ref_motion_state)

    joint_pos = ref_motion_state.joint_pos[env_ids]
    joint_vel = ref_motion_state.joint_vel[env_ids]
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env._joint_ids, env_ids=env_ids)

    root_states = asset.data.default_root_state[env_ids].clone()

    root_states[:, :3] = ref_motion_state.root_pos[env_ids]
    root_states[:, 2] += 0.04  # in case under the terrain

    root_states[:, 3:7] = ref_motion_state.root_rot[env_ids]
    root_states[:, 7:10] = ref_motion_state.root_lin_vel[env_ids]
    root_states[:, 10:13] = ref_motion_state.root_ang_vel[env_ids]

    state = root_states[:, :7]
    velocities = root_states[:, 7:13]
    asset.write_root_pose_to_sim(state, env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def update_curriculum(
    env: NeuralWBCEnv,
    env_ids: torch.Tensor | None,
    penalty_level_down_threshold: float,
    penalty_level_up_threshold: float,
    penalty_level_degree: float,
    min_penalty_scale: float,
    max_penalty_scale: float,
    num_compute_average_epl: float,
):
    """
    Update average episode length and in turn penalty curriculum.

    This function is rewritten from update_average_episode_length of legged_gym.

    When the policy is not able to track the motions, we reduce the penalty to help it explore more actions. When the
    policy is able to track the motions, we increase the penalty to smooth the actions and reduce the maximum action
    it uses.
    """
    N = env.num_envs if env_ids is None else len(env_ids)
    current_average_episode_length = torch.mean(env.episode_length_buf[env_ids], dtype=torch.float)
    env.average_episode_length = env.average_episode_length * (
        1 - N / num_compute_average_epl
    ) + current_average_episode_length * (N / num_compute_average_epl)

    if env.average_episode_length < penalty_level_down_threshold:
        env.penalty_scale *= 1 - penalty_level_degree
    elif env.average_episode_length > penalty_level_up_threshold:
        env.penalty_scale *= 1 + penalty_level_degree
    env.penalty_scale = np.clip(env.penalty_scale, min_penalty_scale, max_penalty_scale)


def cache_body_mass_scale(
    env: NeuralWBCEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses()
    scale = (masses / asset.data.default_mass).to(env.device)

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if not hasattr(env, "body_mass_scale"):
        # Cache body mass randomization scale for privileged observation.
        env.mass_randomized_body_ids, _ = asset.find_bodies(env.cfg.mass_randomized_body_names, preserve_order=True)
        env.body_mass_scale = torch.ones(
            env.num_envs, len(env.mass_randomized_body_ids), dtype=torch.float, device=env.device
        )

    env.body_mass_scale[env_ids, :] *= scale[env_ids[:, None], env.mass_randomized_body_ids]


def resolve_dist_fn(
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    dist_fn = math_utils.sample_uniform

    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise ValueError(f"Unrecognized distribution {distribution}")

    return dist_fn
