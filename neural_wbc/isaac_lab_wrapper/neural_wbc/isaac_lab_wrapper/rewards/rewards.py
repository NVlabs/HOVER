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


from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from neural_wbc.core import math_utils

from isaaclab.assets import ArticulationData
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from ._env import NeuralWBCEnv

import phc.utils.torch_utils as torch_utils

from neural_wbc.core.body_state import BodyState
from neural_wbc.core.reference_motion import ReferenceMotionState

from .reward_cfg import NeuralWBCRewardCfg


class NeuralWBCRewards:
    """
    Convenience class for computing rewards.
    """

    def __init__(
        self,
        env: NeuralWBCEnv,
        reward_cfg: NeuralWBCRewardCfg,
        contact_sensor: ContactSensor,
        contact_sensor_feet_ids: list,
        body_state_feet_ids: list,
    ):
        self._num_envs = env.num_envs
        self._device = env.device
        self._cfg = reward_cfg
        self._torque_limits = torch.tensor(self._cfg.torque_limits, device=self._device) * self._cfg.torque_limits_scale
        self._joint_pos_limits = torch.tensor(self._cfg.joint_pos_limits, device=self._device)
        self._joint_vel_limits = (
            torch.tensor(self._cfg.joint_vel_limits, device=self._device) * self._cfg.joint_vel_limits_scale
        )
        self.contact_sensor = contact_sensor
        self.contact_sensor_feet_ids = contact_sensor_feet_ids
        self._body_state_feet_ids = body_state_feet_ids
        self._dt = env.step_dt

        self._joint_id_reorder = env._joint_ids

        # In the original implementation in Isaac Gym, the direction of gravity can be configured with an "up axis"
        # parameter. In Isaac Sim, "up axis" is always Z.
        self._gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self._device).repeat((self._num_envs, 1))
        self._feet_max_height_in_air = torch.zeros(
            self._num_envs, len(self.contact_sensor_feet_ids), dtype=torch.float, device=self._device
        )

    def compute_reward(
        self,
        articulation_data: ArticulationData,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        previous_actions: torch.Tensor,
        actions: torch.Tensor,
        reset_buf: torch.Tensor,
        timeout_buf: torch.Tensor,
        penalty_scale: float,
    ) -> tuple[torch.Tensor, dict]:
        """
        Computes the total reward for the given environment and reference motion state.

        This function calculates the weighted sum of individual rewards specified in the environment's
        reward configuration. Each reward is computed using a corresponding reward function defined
        within the class. The final reward is a sum of these individual rewards, each scaled by a
        specified factor.

        Returns:
            tuple[torch.Tensor, dict]: A tuple containing:
                - reward_sum (torch.Tensor): The total computed reward for all environments.
                - rewards (dict): A dictionary with individual reward names as keys and their computed values as values.

        Raises:
            AttributeError: If a reward function corresponding to a reward name is not defined.
        """
        reward_sum = torch.zeros([self._num_envs], device=self._device)
        rewards = {}
        for reward_name, scale in self._cfg.scales.items():
            try:
                reward_fn = getattr(self, reward_name)
            except AttributeError:
                raise AttributeError(f"No reward or penalty function is defined for {reward_name}")

            rewards[reward_name] = reward_fn(
                body_state=body_state,
                ref_motion_state=ref_motion_state,
                articulation_data=articulation_data,
                previous_actions=previous_actions,
                actions=actions,
                reset_buf=reset_buf,
                timeout_buf=timeout_buf,
            )
            if reward_name.startswith("penalize"):
                rewards[reward_name] *= penalty_scale
            reward_sum += rewards[reward_name] * scale

        return reward_sum, rewards

    def reward_track_joint_positions(
        self,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the reward for tracking the joint positions of the reference motion.

        This function is rewritten from _reward_teleop_selected_joint_position of legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed reward for each environment.
        """
        joint_pos = body_state.joint_pos
        ref_joint_pos = ref_motion_state.joint_pos
        mean_joint_pos_diff_squared = torch.mean(torch.square(ref_joint_pos - joint_pos), dim=1)
        return torch.exp(-mean_joint_pos_diff_squared / self._cfg.joint_pos_sigma)

    def reward_track_joint_velocities(
        self,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the reward for tracking the joint velocities of the reference motion.

        This function is rewritten from _reward_teleop_selected_joint_vel of legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed reward for each environment.
        """
        joint_vel = body_state.joint_vel
        ref_joint_vel = ref_motion_state.joint_vel
        mean_joint_vel_diff_squared = torch.mean(torch.square(ref_joint_vel - joint_vel), dim=1)
        return torch.exp(-mean_joint_vel_diff_squared / self._cfg.joint_vel_sigma)

    def reward_track_body_velocities(
        self,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes a reward based on the difference between the body's velocity and the reference motion's velocity.

        This function is rewritten from _reward_teleop_body_vel of legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed reward for each environment.
        """
        body_vel = body_state.body_lin_vel
        ref_body_vel = ref_motion_state.body_lin_vel
        diff_vel = ref_body_vel - body_vel
        mean_diff_vel_squared = (diff_vel**2).mean(dim=-1).mean(dim=-1)
        return torch.exp(-mean_diff_vel_squared / self._cfg.body_vel_sigma)

    def reward_track_body_angular_velocities(
        self,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes a reward based on the difference between the body's angular velocity and the reference motion's angular
        velocity.

        This function is rewritten from _reward_teleop_body_ang_vel of legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed reward for each environment.
        """
        body_ang_vel = body_state.body_ang_vel
        ref_body_ang_vel = ref_motion_state.body_ang_vel
        diff_ang_vel = ref_body_ang_vel - body_ang_vel
        mean_diff_ang_vel_squared = (diff_ang_vel**2).mean(dim=-1).mean(dim=-1)
        return torch.exp(-mean_diff_ang_vel_squared / self._cfg.body_ang_vel_sigma)

    def reward_track_body_position_extended(
        self,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes a reward based on the difference between the body's extended position and the reference motion's
        extended position.

        This function is rewritten from _reward_teleop_body_position_extend of legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed reward for each environment.
        """
        body_pos_extend = body_state.body_pos_extend
        ref_body_pos_extend = ref_motion_state.body_pos_extend

        diff_global_body_pos = ref_body_pos_extend - body_pos_extend
        diff_global_body_pos_lower = diff_global_body_pos[:, :11]
        diff_global_body_pos_upper = diff_global_body_pos[:, 11:]
        diff_body_pos_dist_lower = (diff_global_body_pos_lower**2).mean(dim=-1).mean(dim=-1)
        diff_body_pos_dist_upper = (diff_global_body_pos_upper**2).mean(dim=-1).mean(dim=-1)
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self._cfg.body_pos_lower_body_sigma)
        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self._cfg.body_pos_upper_body_sigma)

        return (
            r_body_pos_lower * self._cfg.body_pos_lower_body_weight
            + r_body_pos_upper * self._cfg.body_pos_upper_body_weight
        )

    def reward_track_body_position_vr_key_points(
        self,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes a reward based on the difference between selected key points of the body's extended position
        and the reference motion's extended position.

        This function is rewritten from _reward_teleop_body_position_vr_3keypoints of legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed reward for each environment.
        """
        body_pos_extend = body_state.body_pos_extend
        ref_body_pos_extend = ref_motion_state.body_pos_extend

        diff_global_body_pos = ref_body_pos_extend - body_pos_extend
        diff_global_body_pos_vr_key_points = diff_global_body_pos[:, -3:]
        diff_body_pos_dist_vr_key_points = (diff_global_body_pos_vr_key_points**2).mean(dim=-1).mean(dim=-1)

        return torch.exp(-diff_body_pos_dist_vr_key_points / self._cfg.body_pos_vr_key_points_sigma)

    def reward_track_body_rotation(
        self,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes a reward based on the difference between the body's rotation and the reference motion's rotation.

        This function is rewritten from _reward_teleop_body_rotation of legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed reward for each environment.
        """
        body_rot = body_state.body_rot
        ref_body_rot = ref_motion_state.body_rot

        diff_global_body_rot = math_utils.quat_mul(ref_body_rot, math_utils.quat_conjugate(body_rot))
        diff_global_body_rot_xyzw = math_utils.convert_quat(diff_global_body_rot, to="xyzw")
        diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot_xyzw)[0]
        diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)

        return torch.exp(-diff_global_body_angle_dist / self._cfg.body_rot_sigma)

    def penalize_torques(
        self,
        articulation_data: ArticulationData,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the penalty on applied torques to minimize energy consumption.

        This function is adapted from _reward_torques in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        return torch.sum(torch.square(articulation_data.applied_torque), dim=1)

    def penalize_by_torque_limits(
        self,
        articulation_data: ArticulationData,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the penalty on actions that exceed the torque limits defined in the configuration.

        This function is adapted from _reward_torque_limits in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        return torch.sum(
            (torch.abs(articulation_data.applied_torque[:, self._joint_id_reorder]) - self._torque_limits).clip(
                min=0.0
            ),
            dim=1,
        )

    def penalize_joint_accelerations(
        self,
        articulation_data: ArticulationData,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the penalty on joint acceleration of each motor.

        This function is adapted from _reward_dof_acc in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        return torch.sum(torch.square(articulation_data.joint_acc), dim=1)

    def penalize_joint_velocities(
        self,
        articulation_data: ArticulationData,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the penalty on joint velocity of each motor.

        This function is adapted from _reward_dof_vel in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        return torch.sum(torch.square(articulation_data.joint_vel), dim=1)

    def penalize_lower_body_action_changes(
        self,
        previous_actions: torch.Tensor,
        actions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the penalty for action changes in the lower body.

        This function is adapted from _reward_lower_action_rate in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        # Joints 0 - 10 are lower body joints in Isaac Gym.
        return torch.sum(torch.square(previous_actions[:, :11] - actions[:, :11]), dim=1)

    def penalize_upper_body_action_changes(
        self,
        previous_actions: torch.Tensor,
        actions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the penalty for action changes in the upper body.

        This function is adapted from _reward_upper_action_rate in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        # Joints 11 - 19 are upper body joints in Isaac Gym.
        return torch.sum(torch.square(previous_actions[:, 11:] - actions[:, 11:]), dim=1)

    def penalize_by_joint_pos_limits(
        self,
        body_state: BodyState,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the penalty on actions that exceed the joint position limits defined in the configuration.

        This function is adapted from _reward_dof_pos_limits in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        joint_pos = body_state.joint_pos
        out_of_bounds = -(joint_pos - self._joint_pos_limits[:, 0]).clip(max=0.0)
        out_of_bounds += (joint_pos - self._joint_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_bounds, dim=1)

    def penalize_by_joint_velocity_limits(
        self,
        body_state: BodyState,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the penalty on actions that exceed the joint velocity limits defined in the configuration.

        This function is adapted from _reward_dof_vel_limits in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        return torch.sum(
            (torch.abs(body_state.joint_vel) - self._joint_vel_limits).clip(min=0.0, max=1.0),
            dim=1,
        )

    def penalize_early_termination(self, reset_buf: torch.Tensor, timeout_buf: torch.Tensor, **kwargs):
        """
        Computes the penalty for episodes that terminate before timeout.

        This function is adapted from `_reward_termination` in `legged_gym`.

        Returns:
            torch.Tensor: A tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        # Terminal reward / penalty
        return (reset_buf * ~timeout_buf).float()

    def penalize_feet_contact_forces(self, **kwargs):
        """
        Computes the penalty on feet contact forces that exceed limits.

        This function is adapted from _reward_feet_contact_forces in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        feet_contact_forces = self._get_feet_contact_forces()
        return torch.sum((torch.norm(feet_contact_forces, dim=-1) - self._cfg.max_contact_force).clip(min=0.0), dim=1)

    def penalize_stumble(self, **kwargs):
        """
        Computes the penalty for stumbling.

        This function is adapted from _reward_stumble in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        feet_contact_forces = self._get_feet_contact_forces()
        return torch.any(
            torch.norm(feet_contact_forces[:, :, :2], dim=2) > 5 * torch.abs(feet_contact_forces[:, :, 2]), dim=1
        ).float()

    def penalize_slippage(self, body_state: BodyState, **kwargs):
        """
        Computes the penalty for slippage.

        This function is adapted from _reward_slippage in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        feet_vel = body_state.body_lin_vel[:, self._body_state_feet_ids]
        feet_contact_forces = self._get_feet_contact_forces()
        return torch.sum(torch.norm(feet_vel, dim=-1) * (torch.norm(feet_contact_forces, dim=-1) > 1.0), dim=1)

    def penalize_feet_orientation(self, body_state: BodyState, **kwargs):
        """
        Computes the penalty on feet orientation to make no x and y projected gravity.

        This function is adapted from _reward_feet_ori in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        left_quat = body_state.body_rot[:, self._body_state_feet_ids[0]]
        left_gravity = math_utils.quat_rotate_inverse(left_quat, self._gravity_vec)
        right_quat = body_state.body_rot[:, self._body_state_feet_ids[1]]
        right_gravity = math_utils.quat_rotate_inverse(right_quat, self._gravity_vec)
        return (
            torch.sum(torch.square(left_gravity[:, :2]), dim=1) ** 0.5
            + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5
        )

    def penalize_feet_air_time(self, ref_motion_state: ReferenceMotionState, **kwargs):
        """
        Computes the penalty for the time that the feet are in the air before the newest contact with the terrain.

        This function is adapted from _reward_feet_air_time_teleop in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        ref_pelvis_vel_xy = ref_motion_state.body_lin_vel[:, 0, :2]
        first_contact = self._get_feet_first_contact()
        last_feet_air_time = self._get_last_air_time_for_feet()
        reward = torch.sum(
            (last_feet_air_time - 0.25) * first_contact, dim=1
        )  # reward only on first contact with the ground
        reward *= torch.norm(ref_pelvis_vel_xy, dim=1) > 0.1  # no reward for low ref motion velocity (root xy velocity)
        return reward

    def penalize_both_feet_in_air(self, **kwargs):
        """
        Computes the penalty for both feet being in the air.

        This function is adapted from _reward_in_the_air in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        feet_in_air = self._get_feet_in_the_air()
        return torch.all(feet_in_air, dim=1).float()

    def penalize_orientation(self, articulation_data: ArticulationData, **kwarg):
        """
        Computes the penalty based on a non-flat base orientation.

        This function is adapted from _reward_orientation in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        projected_gravity = articulation_data.projected_gravity_b
        return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

    def penalize_max_feet_height_before_contact(self, body_state: BodyState, **kwargs):
        """
        Computes the penalty based on the maximum height of the feet in the air before the current contact.

        This function is adapted from _reward_feet_max_height_for_this_air in legged_gym.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
        """
        first_contact = self._get_feet_first_contact()
        feet_height = body_state.body_pos[:, self._body_state_feet_ids, 2]
        self._feet_max_height_in_air = torch.max(self._feet_max_height_in_air, feet_height)
        feet_max_height = torch.sum(
            (torch.clamp_min(self._cfg.max_feet_height_limit_before_contact - self._feet_max_height_in_air, 0))
            * first_contact,
            dim=1,
        )  # reward only on first contact with the ground
        feet_in_air = self._get_feet_in_the_air()
        self._feet_max_height_in_air *= feet_in_air
        return feet_max_height

    def _get_feet_contact_forces(self):
        """Gets the current contact forces on each feet body part.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs, num_feet_body_parts)
        """
        return self.contact_sensor.data.net_forces_w[:, self.contact_sensor_feet_ids, :]

    def _get_last_air_time_for_feet(self):
        """Gets the time in the air before the most recent contact.

        Returns:
            torch.Tensor: A float tensor of shape (num_envs, num_feet_body_parts)
        """
        return self.contact_sensor.data.last_air_time[:, self.contact_sensor_feet_ids]

    def _get_feet_in_the_air(self):
        """Checks whether each feet body part is in the air.

        Returns:
            torch.Tensor: A boolean tensor of shape (num_envs, num_feet_body_parts)
        """
        return self._get_feet_contact_forces()[:, :, 2] <= 1.0

    def _get_feet_first_contact(self):
        """Checks whether each feet body part has moved from the air (no contact) to a surface (contact)
        in the most recent step.

        Returns:
            torch.Tensor: A boolean tensor of shape (num_envs, num_feet_body_parts)
        """
        return self.contact_sensor.compute_first_contact(self._dt)[:, self.contact_sensor_feet_ids]
