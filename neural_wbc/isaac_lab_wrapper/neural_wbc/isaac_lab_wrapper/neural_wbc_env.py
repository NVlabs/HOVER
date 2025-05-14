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

import pprint
import torch
from typing import TYPE_CHECKING

from neural_wbc.core import ReferenceMotionManager, ReferenceMotionState, mask
from neural_wbc.core.observations import StudentHistory
from neural_wbc.core.termination import check_termination_conditions
from neural_wbc.isaac_lab_wrapper.control import resolve_control_fn
from neural_wbc.isaac_lab_wrapper.observations import compute_observations

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.noise import NoiseModel, NoiseModelCfg, UniformNoiseCfg

from .body_state import build_body_state
from .rewards import NeuralWBCRewards
from .visualization import RefMotionVisualizer

if TYPE_CHECKING:
    from .env_cfg import NeuralWBCEnvCfg


def print_config(info: str, names: list[str], values: list | torch.Tensor):
    """Print named configuration values."""
    print(f"[INFO] {info}:")
    pprint.pprint(dict(zip(names, values)), indent=4, sort_dicts=False)


class NeuralWBCEnv(DirectRLEnv):
    cfg: NeuralWBCEnvCfg

    def __init__(self, cfg: NeuralWBCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_actions = self.action_space.shape[1]
        self.num_observations = self.observation_space.shape[1]

        # Joint position command (deviation from default joint positions)
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)

        # Resolve the joints on which the actions are applied.
        self._joint_ids, self._joint_names = self._robot.find_joints(
            self.cfg.joint_names,
            preserve_order=True,
        )
        print_config("self._joint_ids", self._joint_names, self._joint_ids)

        # Resolve the bodies.
        self._body_ids, self._body_names = self._robot.find_bodies(
            self.cfg.body_names,
            preserve_order=True,
        )
        print_config("self._body_ids", self._body_names, self._body_ids)

        # Resolve the extended bodies
        self._body_names_extend = self._body_names + self.cfg.extend_body_names
        additional_ids = list(range(len(self._body_names), len(self._body_names_extend)))
        self._body_ids_extend = self._body_ids + additional_ids
        print_config("self._body_ids_extend", self._body_names_extend, self._body_ids_extend)

        if self.cfg.extend_body_parent_names:
            _, parent_names = self._robot.find_bodies(
                self.cfg.extend_body_parent_names,
                preserve_order=True,
            )
            self.extend_body_parent_ids = [self._body_names.index(n) for n in parent_names]
            self.extend_body_pos = self.cfg.extend_body_pos.repeat(self.num_envs, 1, 1).to(self.device)

        # Get specific body indices
        self.base_id, self.base_name = self.contact_sensor.find_bodies(self.cfg.base_name)
        self.base_id = self.base_id[0]
        self.base_name = self.base_name[0]
        print_config("self._base_id", [self.base_name], [self.base_id])

        self.feet_ids, self.feet_names = self.contact_sensor.find_bodies(self.cfg.feet_name)
        print_config("self.feet_ids", self.feet_names, self.feet_ids)

        self._undesired_contact_body_ids, undesired_contact_body_name = self.contact_sensor.find_bodies(
            self.cfg.undesired_contact_body_names
        )
        print_config(
            "self._undesired_contact_body_ids",
            undesired_contact_body_name,
            self._undesired_contact_body_ids,
        )

        # resolve the pd gain for each joint
        self._p_gains = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)
        self._d_gains = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)

        self.kp_scale = torch.ones((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)
        self.kd_scale = torch.ones((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)

        self.default_kp_scale = self.kp_scale.clone()
        self.default_kd_scale = self.kd_scale.clone()

        for key, value in self.cfg.stiffness.items():
            joint_ids, joint_names = self._robot.find_joints(key, preserve_order=True)
            self._p_gains[:, joint_ids] = value
        for key, value in self.cfg.damping.items():
            joint_ids, joint_names = self._robot.find_joints(key, preserve_order=True)
            self._d_gains[:, joint_ids] = value
        self._p_gains = self._p_gains[:, self._joint_ids]
        self._d_gains = self._d_gains[:, self._joint_ids]

        print_config("self._p_gains", self._joint_names, self._p_gains[0])
        print_config("self._d_gains", self._joint_names, self._d_gains[0])

        # resolve the controller
        self._control_fn = resolve_control_fn(self.cfg.control_type)

        # resolve the control delay
        self.action_queue = torch.zeros(
            (self.num_envs, self.cfg.ctrl_delay_step_range[1] + 1, self.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self._action_delay = torch.randint(
            self.cfg.ctrl_delay_step_range[0],
            self.cfg.ctrl_delay_step_range[1] + 1,
            (self.num_envs,),
            device=self.device,
            requires_grad=False,
        )

        # resolve the control noise: we will add noise to the final torques. the _rfi_lim defines
        # the sample range of the added noise. It represented by the percentage of the control limits.
        # noise = uniform(self.rfi_lim*joint_effort_limit, self.rfi_lim_*joint_effort_limit)
        self.default_rfi_lim = self.cfg.default_rfi_lim * torch.ones(
            (self.num_envs, self.num_actions), dtype=torch.float, device=self.device
        )
        self.rfi_lim = self.default_rfi_lim.clone()

        # resolve torque_limits (the order here is IsaacLab order, have not reordered by cfg.)
        self.torque_limits = torch.ones((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)
        for actuator in self._robot.actuators.values():
            actuator_joint_ids = actuator.joint_indices
            actuator_torque_limits = actuator.effort_limit
            self.torque_limits[:, actuator_joint_ids] *= actuator_torque_limits
        print("[INFO]: self.torque_limits", self._robot.joint_names, self.torque_limits[0])

        # Randomize robot base com
        if not hasattr(self, "default_coms"):
            self.default_coms = self._robot.root_physx_view.get_coms().clone()
            self.base_com_bias = torch.zeros((self.num_envs, 3), dtype=torch.float, device="cpu")

        # Initialize recovery counters
        self.recovery_counters = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # Observations
        # For the teacher we use `self._tracked_body_ids` to define which states should be tracked.
        self._tracked_body_ids = [self._body_names_extend.index(i) for i in self.cfg.tracked_body_names]

        # For the student we use a mask to define which states should be tracked.
        self.mask_element_names = mask.create_mask_element_names(
            body_names=self._body_names_extend,
            joint_names=self._joint_names,
        )
        self._mask = torch.ones((self.num_envs, len(self.mask_element_names)), dtype=torch.bool, device=self.device)
        self._reset_mask()

        if self.cfg.add_policy_obs_noise:
            self._policy_observation_noise_scale_vec = self._get_policy_observation_noise_scale()
            print(
                "[INFO]: self._policy_observation_noise_scale_vec",
                self._policy_observation_noise_scale_vec,
                self._policy_observation_noise_scale_vec.shape,
            )
            observation_noise_model_cfg = NoiseModelCfg(
                noise_cfg=UniformNoiseCfg(
                    n_min=-self._policy_observation_noise_scale_vec,
                    n_max=self._policy_observation_noise_scale_vec,
                    operation="add",
                )
            )
            self._observation_noise_model: NoiseModel = observation_noise_model_cfg.class_type(
                observation_noise_model_cfg, num_envs=self.num_envs, device=self.device
            )

        # Cache body mass randomization scale for privileged observation.
        if not hasattr(self, "body_mass_scale"):
            self.mass_randomized_body_ids, _ = self._robot.find_bodies(
                self.cfg.mass_randomized_body_names, preserve_order=True
            )
            self.body_mass_scale = torch.ones(
                self.num_envs, len(self.mass_randomized_body_ids), dtype=torch.float, device=self.device
            )

        # Logging reward sums
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in self.cfg.rewards.scales
        }

        # Load reference motion
        self._ref_motion_mgr = ReferenceMotionManager(
            cfg=self.cfg.reference_motion_manager,
            device=self.device,
            num_envs=self.num_envs,
            random_sample=(self.cfg.mode.is_training_mode()),
            extend_head=True,
            dt=self.cfg.decimation * self.cfg.dt,
        )
        self.ref_episodic_offset = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.default_ref_episodic_offset = self.ref_episodic_offset.clone()

        # Variables for Resample reference motion
        self.resample_motions_for_envs_interval = self.cfg.resample_motions_for_envs_interval_s // (
            self.cfg.decimation * self.cfg.dt
        )
        self.is_resample_reference_motion_step = False

        # Visualizations
        self._ref_motion_visualizer = RefMotionVisualizer()
        # Start positions of each environment
        self._start_positions_on_terrain = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        # Rewards
        feet_body_ids = [self._body_names.index(feet_name) for feet_name in self.feet_names]
        self._rewards = NeuralWBCRewards(
            env=self,
            reward_cfg=self.cfg.rewards,
            contact_sensor=self.contact_sensor,
            contact_sensor_feet_ids=self.feet_ids,
            body_state_feet_ids=feet_body_ids,
        )
        self._termination_conditions = {}

        # Curriculum
        self.average_episode_length = 0
        self.penalty_scale = 0.5

        # Distill
        if self.cfg.mode.is_distill_mode():
            self.history = StudentHistory(
                num_envs=self.num_envs,
                device=self.device,
                entry_length=self.cfg.single_history_dim,
                max_entries=self.cfg.observation_history_length,
            )

            # Proprioceptive states are joint pos, joint vel, IMU acc, IMU gyro
            num_proprioceptive_states = 2 * len(self._joint_ids) + 6
            self.num_student_obs = (
                # Last proprioceptive measurements:
                num_proprioceptive_states
                # Masked reference motion error:
                + mask.calculate_command_length(
                    num_bodies=len(self._body_names_extend),
                    num_joints=len(self._joint_ids),
                )
                # Mask:
                + len(self.mask_element_names)
                # Last actions:
                + self.num_actions
                # History with proprioceptive states and actions:
                + self.cfg.observation_history_length * (num_proprioceptive_states + self.num_actions)
            )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_mask(self, env_ids: torch.Tensor | None = None):
        """
        Reset the mask used to select which parts of the reference state should
        be tracked. Every environment uses a different mask.
        """
        if self.cfg.mode.is_distill_mode() or self.cfg.mode.is_distill_test_mode():
            if env_ids is None or len(env_ids) == self.num_envs:
                env_ids = self._robot._ALL_INDICES
            # Masks are only needed for distillation and testing.
            self._mask[env_ids] = mask.create_mask(
                mask_element_names=self.mask_element_names,
                mask_modes=self.cfg.distill_mask_modes,
                enable_sparsity_randomization=self.cfg.distill_mask_sparsity_randomization_enabled,
                device=self.device,
                num_envs=len(env_ids),
            )

    def _pre_physics_step(self, actions: torch.Tensor):
        # Update recovery counters
        self.recovery_counters -= 1
        self.recovery_counters = self.recovery_counters.clip(min=0)

        # update history
        if self.cfg.mode.is_distill_mode():
            obs_dic = self._get_observations()
            self.history.update(obs_dic)

        # Action delay process
        if self.cfg.ctrl_delay_step_range[1] > 0:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions.clone()
            self.actions = self.action_queue[torch.arange(self.num_envs), self._action_delay].clone()
        else:
            self.actions = actions.clone()

    def _apply_action(self):
        # We define our own control law that can update at every physics step. Therefore we put
        # this computation here.
        actions_scaled = self.actions * self.cfg.action_scale
        self._processed_actions = self._control_fn(self, actions_scaled, joint_ids=self._joint_ids)

        # Adding noise to the control signal to enhance robustness. `torque_limits` is using IsaacLab
        # ordering, therefore we need to reorder the noise.
        actions_noise = (
            (torch.rand_like(self._processed_actions) * 2.0 - 1.0)
            * self.rfi_lim
            * self.torque_limits[:, self._joint_ids]
        )
        self._processed_actions += actions_noise

        self._robot.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        self._previous_actions = self.actions.clone()

        body_state = build_body_state(
            data=self._robot.data,
            root_id=self.cfg.root_id,
            body_ids=self._body_ids,
            joint_ids=self._joint_ids,
            extend_body_parent_ids=self.extend_body_parent_ids,
            extend_body_pos=self.extend_body_pos,
        )
        ref_motion_state: ReferenceMotionState = self._ref_motion_mgr.get_state_from_motion_lib_cache(
            episode_length_buf=self.episode_length_buf,
            offset=self._start_positions_on_terrain,
            terrain_heights=self.get_terrain_heights(),
        )
        self._ref_motion_visualizer.visualize(ref_motion_state, self._mask)

        # Add additional data that is useful for evaluation.
        self.extras["data"] = {
            "mask": self._mask.detach().clone(),
            "state": {
                "body_pos": body_state.body_pos_extend.detach().clone(),
                "joint_pos": body_state.joint_pos.detach().clone(),
                "root_pos": body_state.root_pos.detach().clone(),
                "root_lin_vel": body_state.root_lin_vel.detach().clone(),
                "root_rot": body_state.root_rot.detach().clone(),
            },
            "ground_truth": {
                "body_pos": ref_motion_state.body_pos_extend.detach().clone(),
                "joint_pos": ref_motion_state.joint_pos.detach().clone(),
                "root_pos": ref_motion_state.root_pos.detach().clone(),
                "root_lin_vel": ref_motion_state.root_lin_vel.detach().clone(),
                "root_rot": ref_motion_state.root_rot.detach().clone(),
            },
            "upper_joint_ids": self.cfg.upper_body_joint_ids,
            "lower_joint_ids": self.cfg.lower_body_joint_ids,
        }

        obs_dic = compute_observations(
            env=self,
            body_state=body_state,
            ref_motion_state=ref_motion_state,
        )

        if self.cfg.add_policy_obs_noise:
            obs_dic["teacher"] = self._observation_noise_model.apply(obs_dic["teacher"])

        return obs_dic

    def _get_rewards(self) -> torch.Tensor:
        ref_motion_state: ReferenceMotionState = self._ref_motion_mgr.get_state_from_motion_lib_cache(
            episode_length_buf=self.episode_length_buf,
            offset=self._start_positions_on_terrain,
            terrain_heights=self.get_terrain_heights(),
        )
        body_state = build_body_state(
            data=self._robot.data,
            body_ids=self._body_ids,
            joint_ids=self._joint_ids,
            extend_body_parent_ids=self.extend_body_parent_ids,
            extend_body_pos=self.extend_body_pos,
            root_id=self.cfg.root_id,
        )

        timeout_buf = (
            self._termination_conditions["reference_motion_length"].reshape(self.num_envs) | self.reset_time_outs
        )
        reward_sum, rewards = self._rewards.compute_reward(
            body_state=body_state,
            ref_motion_state=ref_motion_state,
            articulation_data=self._robot.data,
            previous_actions=self._previous_actions,
            actions=self.actions,
            reset_buf=self.reset_buf,  # From DirectRLEnv
            timeout_buf=timeout_buf,
            penalty_scale=self.penalty_scale,
        )

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward_sum

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        ref_motion_state = self._ref_motion_mgr.get_state_from_motion_lib_cache(
            episode_length_buf=self.episode_length_buf,
            offset=self._start_positions_on_terrain,
            terrain_heights=self.get_terrain_heights(),
        )
        body_state = build_body_state(
            data=self._robot.data,
            # Though this creates a biased condition during training, we find it is helpful for getting a good policy.
            body_ids=self._body_ids,
            joint_ids=self._joint_ids,
            extend_body_parent_ids=self.extend_body_parent_ids,
            extend_body_pos=self.extend_body_pos,
            root_id=self.cfg.root_id,
        )
        died, self._termination_conditions = check_termination_conditions(
            training_mode=self.cfg.mode.is_training_mode(),
            body_state=body_state,
            ref_motion_state=ref_motion_state,
            projected_gravity=self._robot.data.projected_gravity_b,
            gravity_x_threshold=self.cfg.gravity_x_threshold,
            gravity_y_threshold=self.cfg.gravity_y_threshold,
            ref_motion_mgr=self._ref_motion_mgr,
            episode_times=self._get_episode_times(),
            max_ref_motion_dist=self.cfg.max_ref_motion_dist,
            in_recovery=self.recovery_counters.unsqueeze(1) != 0,
            mask=self._mask,
            # We only need the last net contact forces.
            net_contact_forces=self.contact_sensor.data.net_forces_w_history[:, 0, :, :],
            undesired_contact_body_ids=self._undesired_contact_body_ids,
        )

        self.extras["termination_conditions"] = self._termination_conditions

        self.is_resample_reference_motion_step = (
            self.common_step_counter % self.resample_motions_for_envs_interval == 0
        ) and self.cfg.resample_motions

        time_out |= self.is_resample_reference_motion_step

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)

        if self.is_resample_reference_motion_step:
            assert torch.equal(env_ids, self._robot._ALL_INDICES)
            self._ref_motion_mgr.load_motions(random_sample=True, start_idx=0)
        super()._reset_idx(env_ids)

        # reset actions
        self.actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.action_queue[env_ids] = 0.0
        self._action_delay[env_ids] = torch.randint(
            self.cfg.ctrl_delay_step_range[0],
            self.cfg.ctrl_delay_step_range[1] + 1,
            (len(env_ids),),
            device=self.device,
            requires_grad=False,
        )

        # reset recovery counter
        self.recovery_counters[env_ids] = 0.0

        # reset history
        if self.cfg.mode.is_distill_mode():
            self.history.reset(env_ids=env_ids)

        # reset mask
        if self.cfg.reset_mask:
            self._reset_mask(env_ids)

        # Logging
        extras = dict()

        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0 * self._episode_sums[key][env_ids]

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    # Properties
    @property
    def reference_motion_manager(self) -> ReferenceMotionManager:
        return self._ref_motion_mgr

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    # Utility functions
    def _get_episode_times(self):
        return self.episode_length_buf * self.cfg.decimation * self.cfg.dt

    def _get_policy_observation_noise_scale(self):
        noise_vec = torch.zeros(self.num_observations, device=self.device)
        observation_body_num = len(self._tracked_body_ids)
        curr_obs_len = 0

        noise_config = [
            ("body_pos", (observation_body_num - 1) * 3),
            ("body_rot", observation_body_num * 6),
            ("body_lin_vel", observation_body_num * 3),
            ("body_ang_vel", observation_body_num * 3),
            ("ref_body_pos_diff", observation_body_num * 3),
            ("ref_body_rot_diff", observation_body_num * 6),
            ("ref_lin_vel", observation_body_num * 3),
            ("ref_ang_vel", observation_body_num * 3),
            ("ref_body_pos", observation_body_num * 3),
            ("ref_body_rot", observation_body_num * 6),
        ]

        for key, length in noise_config:
            noise_vec[curr_obs_len : curr_obs_len + length] = (
                self.cfg.policy_obs_noise_scales[key] * self.cfg.policy_obs_noise_level
            )
            curr_obs_len += length

        return noise_vec

    def get_terrain_heights(self) -> torch.Tensor:
        return self._height_scanner.data.ray_hits_w.clone()[..., 2]
