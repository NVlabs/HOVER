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

import math
import torch
from typing import Literal

from inference_env.neural_wbc_env_cfg import NeuralWBCEnvCfg
from mujoco_wrapper.control import resolve_control_fn

from neural_wbc.core import EnvironmentWrapper, ReferenceMotionManager
from neural_wbc.core.body_state import BodyState
from neural_wbc.core.observations import StudentHistory, compute_student_observations
from neural_wbc.core.robot_wrapper import get_robot_class, get_robot_names
from neural_wbc.core.termination import check_termination_conditions


class NeuralWBCEnv(EnvironmentWrapper):
    """Mujoco Neural WBC Environment

    Neural WBC Environment using WBCMujoco for validation of IsaacLab policies
    """

    cfg: NeuralWBCEnvCfg

    def __init__(
        self,
        cfg: NeuralWBCEnvCfg,
        render_mode: str | None = None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        """Initializes the environment.

        Args:
            cfg (NeuralWBCEnvCfgH1): Environment configuration.
            render_mode (str | None, optional): Render mode. Defaults to None.
            device (torch.device, optional): torch device to use. Defaults to 'cuda'.
        """
        super().__init__(mode=cfg.mode)
        self.cfg = cfg
        self.render_mode = render_mode
        self.num_envs = 1
        self._env_ids = torch.arange(0, self.num_envs, device=device)
        self.device = device
        self.reference_motion_manager = ReferenceMotionManager(
            cfg=self.cfg.reference_motion_cfg,
            device=self.device,
            num_envs=self.num_envs,
            random_sample=(self.cfg.mode.is_training_mode()),
            extend_head=True,
            dt=self.cfg.decimation * self.cfg.dt,
        )

        # Start positions of each environment
        self._start_positions_on_terrain = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        if cfg.robot not in get_robot_names():
            raise ValueError(f"Unknown robot: {cfg.robot}, options are: {get_robot_names()}")
        robot_class = get_robot_class(cfg.robot)
        self._robot = robot_class(
            cfg,
            num_instances=1,
            device=self.device,
        )

        self._joint_ids = self._robot.get_joint_ids()
        self._body_ids = self._robot.get_body_ids()
        self._base_name = "torso_link"
        self._base_id = self._body_ids[self._base_name]

        self.num_actions = self._robot.num_controls

        print("[INFO]: Joint ids", self._joint_ids)
        print("[INFO]: Body ids", self._body_ids)

        # actions
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Resolve the extended bodies
        if self.cfg.extend_body_parent_names:
            self.extend_body_parent_ids = [self._body_ids[name] for name in self.cfg.extend_body_parent_names]
            self.extend_body_pos = self.cfg.extend_body_pos.repeat(self.num_envs, 1, 1).to(self.device)
            for name in self.cfg.extend_body_names:
                self._body_ids[name] = len(self._body_ids)
        else:
            self.extend_body_parent_ids = None
            self.extend_body_pos = None

        self._tracked_body_ids = [self._body_ids[name] for name in self.cfg.tracked_body_names]

        # Initialize extras
        self.extras = {}

        # Distill
        if self.cfg.mode.is_distill_mode():
            self.history = StudentHistory(
                num_envs=self.num_envs,
                device=self.device,
                entry_length=self.cfg.single_history_dim,
                max_entries=self.cfg.observation_history_length,
            )

        print("[INFO]: Setting up pd gains")
        self._p_gains = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)
        self._d_gains = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)
        for key, value in self.cfg.stiffness.items():
            joint_id_dict = self._robot.get_joint_ids([key])
            self._p_gains[:, joint_id_dict[key]] = value
            print("[INFO]: Setting up p gains", joint_id_dict[key], key, value)
        for key, value in self.cfg.damping.items():
            joint_id_dict = self._robot.get_joint_ids([key])
            self._d_gains[:, joint_id_dict[key]] = value
            print("[INFO]: Setting up d gains", joint_id_dict[key], key, value)

        self._effort_limit = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)
        for key, value in self.cfg.effort_limit.items():
            joint_id_dict = self._robot.get_joint_ids([key])
            self._effort_limit[:, joint_id_dict[key]] = value
            print("[INFO]: Setting effort limit", joint_id_dict[key], key, value)

        self._lower_position_limit = torch.zeros(
            (self.num_envs, self.num_actions), dtype=torch.float, device=self.device
        )
        self._upper_position_limit = torch.zeros(
            (self.num_envs, self.num_actions), dtype=torch.float, device=self.device
        )
        for key, value in self.cfg.position_limit.items():
            joint_id_dict = self._robot.get_joint_ids([key])
            self._lower_position_limit[:, joint_id_dict[key]] = value[0]
            self._upper_position_limit[:, joint_id_dict[key]] = value[1]
            print("[INFO]: Setting position limit", joint_id_dict[key], key, value)

        # resolve the controller
        self._control_fn = resolve_control_fn(self.cfg.control_type)

        # resolve the limit type
        self._apply_limits = self._resolve_limit_fn(self.cfg.robot_actuation_type)

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

        # default state
        self._default_qpos = self._robot.default_joint_positions
        self._default_qvel = self._robot.default_joint_velocities
        self._update_robot_default_state()

        # The mask is ordered by bodies, joints, root references. Since bodies are first we can
        # directly reuse the self._tracked_body_ids here too.
        self._mask = torch.zeros((self.num_envs, self.cfg.mask_length), device=self.device).bool()
        self._mask[:, self._tracked_body_ids] = True

        self.reset()

    @property
    def robot(self):
        """Get the underlying robot / articulation"""
        return self._robot

    @property
    def max_episode_length(self):
        """The maximum episode length in steps adjusted from s."""
        return math.ceil(self.cfg.max_episode_length_s / (self.cfg.dt * self.cfg.decimation))

    @property
    def p_gains(self) -> torch.Tensor:
        """The P gains for each joint."""
        return self._p_gains

    @property
    def d_gains(self) -> torch.Tensor:
        """The D gains for each joint."""
        return self._d_gains

    @property
    def default_joint_pos(self) -> torch.Tensor:
        """The default joint positions."""
        return self._default_qpos[:, self.robot.joint_pos_offset :]

    def _apply_position_limits(self, actions: torch.Tensor):
        """Applies position limits to the actions."""
        actions = torch.clip(actions, min=self._lower_position_limit, max=self._upper_position_limit)
        return actions

    def _apply_effort_limits(self, actions: torch.Tensor):
        """Applies effort limits to the actions."""
        actions = torch.clip(actions, min=-self._effort_limit, max=self._effort_limit)
        return actions

    def _resolve_limit_fn(
        self,
        robot_actuation_type: Literal["Pos", "Torque"] = "Torque",
    ):
        if robot_actuation_type == "Pos":
            return self._apply_position_limits
        elif robot_actuation_type == "Torque":
            return self._apply_effort_limits
        else:
            raise ValueError(f"Unrecognized robot actuation type {robot_actuation_type}")

    def step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
        """Performs one step of simulation.

        Args:
            actions (torch.Tensor): Actions of shape (num_envs, num_actions)

        Returns:
            * Observations as a dict or an object.
            * Rewards of shape (num_envs,)
            * "Dones": a boolean tensor representing termination of an episode in each environment.
            * Extra information captured, as a dict, always empty.
        """

        # process actions
        self._pre_physics_step(actions)

        for _ in range(self.cfg.decimation):
            self._apply_action()

            # Convert it to numpy and apply it to the simulator.
            processed_action_np = self._processed_action.detach().cpu().numpy()
            self.robot.step(processed_action_np)

        # Forward the current episode step buffer to keep track of current number of steps into the motion reference.
        self.episode_length_buf += 1

        terminated, time_outs = self._get_dones()
        dones = (terminated | time_outs).to(dtype=terminated.dtype)

        # This env will be used for testing only.
        rewards = {}

        obs_dict = self._compute_observations()
        if self.cfg.mode.is_distill_mode():
            obs = obs_dict["student_policy"]
        else:
            obs = obs_dict["teacher"]

        # Extras are required for evaluation
        self.extras["observations"] = obs_dict
        extras = self._compute_extras()
        self.extras.update(extras)

        return rewards, obs, dones, self.extras

    def _pre_physics_step(self, actions: torch.Tensor):
        """Prepares the robot for the next physics step.

        Args:
            actions (torch.Tensor): Actions of shape (num_envs, num_actions)
        """
        # update history
        if self.cfg.mode.is_distill_mode():
            obs_dic = self._compute_observations()
            self.history.update(obs_dic)

        # Action delay process
        if self.cfg.ctrl_delay_step_range[1] > 0:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions.clone()
            self.actions = self.action_queue[torch.arange(self.num_envs), self._action_delay].clone()
        else:
            self.actions = actions.clone()

    def reset(self, env_ids: list | torch.Tensor | None = None):
        """Resets environment specified by env_ids.

        Args:
            env_ids (list | torch.Tensor | None, optional): Environment ids. Defaults to None.
        """
        env_ids = self._env_ids if env_ids is None else env_ids

        self._reset_robot_state_and_motion(env_ids=env_ids)

        # reset actions
        self.actions[env_ids] = 0.0
        self.action_queue[env_ids] = 0.0
        self._action_delay[env_ids] = torch.randint(
            self.cfg.ctrl_delay_step_range[0],
            self.cfg.ctrl_delay_step_range[1] + 1,
            (len(env_ids),),
            device=self.device,
            requires_grad=False,
        )

        # reset history
        if self.cfg.mode.is_distill_mode():
            self.history.reset(env_ids=env_ids)

        obs, _ = self.get_observations()
        return obs, None

    def get_student_observations(self) -> torch.Tensor:
        """Gets student policy observations for each environment.

        Returns:
            torch.Tensor: Student observations
        """
        current_obs = self._compute_observations()
        return current_obs["student_policy"]

    def _reset_robot_state_and_motion(self, env_ids: list | torch.Tensor | None = None):
        """Resets the robot state and the reference motion.

        Args:
            env_ids (list | torch.Tensor | None, optional): Environment ids. Defaults to None.
        """
        env_ids = env_ids if env_ids is not None else self._env_ids
        self.reference_motion_manager.reset_motion_start_times(env_ids=env_ids, sample=False)

        self.episode_length_buf[env_ids] = 0

        # Record new start locations on terrain
        self._start_positions_on_terrain[env_ids, ...] = self._default_qpos[env_ids, :3]

        ref_motion_state = self.reference_motion_manager.get_state_from_motion_lib_cache(
            episode_length_buf=self.episode_length_buf,
            offset=self._start_positions_on_terrain,
            terrain_heights=self.robot.get_terrain_heights(),
        )

        joint_pos = ref_motion_state.joint_pos[env_ids]
        joint_vel = ref_motion_state.joint_vel[env_ids]

        # Note: In IsaacLab implementation, the z direction is offset by a constant, which is not necessary for mujoco.
        root_pos = ref_motion_state.root_pos[env_ids]
        root_quat = ref_motion_state.root_rot[env_ids]

        root_lin_vel = ref_motion_state.root_lin_vel[env_ids]
        root_ang_vel = ref_motion_state.root_ang_vel[env_ids]

        # Assemble the state of generalized coordinates.
        qpos = torch.hstack((root_pos, root_quat, joint_pos))
        qvel = torch.hstack((root_ang_vel, root_lin_vel, joint_vel))

        self.robot.reset(qpos=qpos, qvel=qvel)

    def _update_robot_default_state(self):
        """Loads the default initial state of the robot."""
        # Read the default values from the mujoco data first.
        robot_init_state = self.cfg.robot_init_state
        state_update_dict = {}
        root_pos_offset = 3
        if "base_pos" in robot_init_state:
            self._default_qpos[:, :root_pos_offset] = torch.tensor(
                self.cfg.robot_init_state["base_pos"], device=self.device
            )
            state_update_dict["root_pos"] = self._default_qpos[:, :root_pos_offset]

        if "base_quat" in robot_init_state:
            self._default_qpos[:, root_pos_offset : self._robot.joint_pos_offset] = torch.tensor(
                self.cfg.robot_init_state["base_quat"], device=self.device
            )
            state_update_dict["root_orientation"] = self._default_qpos[
                :, root_pos_offset : self._robot.joint_pos_offset
            ]

        joint_pos = robot_init_state.get("joint_pos", {})
        joint_vel = robot_init_state.get("joint_vel", {})

        for key, value in joint_pos.items():
            joint_id = self._robot.get_joint_ids([key])[key]
            self._default_qpos[:, joint_id + self._robot.joint_pos_offset] = value

        for key, value in joint_vel.items():
            joint_id = self._robot.get_joint_ids([key])[key]
            self._default_qvel[:, joint_id + self._robot.joint_vel_offset] = value

        state_update_dict["joint_positions"] = self._default_qpos[:, self._robot.joint_pos_offset :]
        state_update_dict["joint_velocities"] = self._default_qvel[:, self._robot.joint_vel_offset :]
        self._robot.update(state_update_dict)

    def _apply_action(self):
        """Applies the current action to the robot."""
        actions_scaled = self.actions * self.cfg.action_scale + self.default_joint_pos
        self._processed_action = self._control_fn(self, actions_scaled)
        self._processed_action = self._apply_limits(self._processed_action)

        # Adding noise to the control signal to enhance robustness
        joint_ids = [v for v in self._joint_ids.values()]
        actions_noise = (
            (torch.rand_like(self._processed_actions) * 2.0 - 1.0) * self.rfi_lim * self._effort_limit[:, joint_ids]
        )
        self._processed_actions += actions_noise

    def _compose_body_state(
        self, extend_body_pos: torch.Tensor | None = None, extend_body_parent_ids: list[int] | None = None
    ) -> BodyState:
        """Compose the Body state from the robot/articulation data.

        Args:
            extend_body_pos (torch.Tensor | None, optional): Extended body positions. Defaults to None.
            extend_body_parent_ids (list[int] | None, optional): Extended body parent ids. Defaults to None.

        Returns:
            BodyState: Composed body state
        """
        lin_vel, ang_vel = self.robot.body_velocities
        body_state = BodyState(
            body_pos=self.robot.body_positions,
            body_rot=self.robot.body_rotations,
            body_lin_vel=lin_vel,
            body_ang_vel=ang_vel,
            joint_pos=self.robot.joint_positions,
            joint_vel=self.robot.joint_velocities,
            root_id=self._base_id,
        )

        if (extend_body_pos is not None) and (extend_body_parent_ids is not None):
            body_state.extend_body_states(
                extend_body_pos=extend_body_pos, extend_body_parent_ids=extend_body_parent_ids
            )
        return body_state

    def _compute_observations(self):
        """Compute the observations

        Returns:
            dict: Observations
        """
        self._robot.update({})

        # NOTE: not including the privileged observations so far
        obs_dict = {}
        ref_motion_state = self.reference_motion_manager.get_state_from_motion_lib_cache(
            episode_length_buf=self.episode_length_buf,
            offset=self._start_positions_on_terrain,
            terrain_heights=self.robot.get_terrain_heights(),
        )
        self._robot.visualize(ref_motion_state=ref_motion_state)

        current_body_state = self._compose_body_state(
            extend_body_pos=self.extend_body_pos, extend_body_parent_ids=self.extend_body_parent_ids
        )

        self.extras["data"] = {
            "mask": self._mask.detach().clone(),
            "state": {
                "body_pos": current_body_state.body_pos_extend.detach().clone(),
                "joint_pos": current_body_state.joint_pos.detach().clone(),
                "root_pos": current_body_state.root_pos.detach().clone(),
                "root_rot": current_body_state.root_rot.detach().clone(),
                "root_lin_vel": current_body_state.root_lin_vel.detach().clone(),
            },
            "ground_truth": {
                "body_pos": ref_motion_state.body_pos_extend.detach().clone(),
                "joint_pos": ref_motion_state.joint_pos.detach().clone(),
                "root_pos": ref_motion_state.root_pos.detach().clone(),
                "root_rot": ref_motion_state.root_rot.detach().clone(),
                "root_lin_vel": ref_motion_state.root_lin_vel.detach().clone(),
            },
            "upper_joint_ids": self.cfg.upper_body_joint_ids,
            "lower_joint_ids": self.cfg.lower_body_joint_ids,
        }

        base_gravity = self.robot.get_base_projected_gravity(self._base_name)
        # The angular velocity from the Unitree H1 IMU is already provided in the robot's local frame.
        # However, in MuJoCo, the angular velocity in the local frame must be computed using world frame mjc body data.
        local_base_ang_velocity = (
            self.robot.get_base_angular_velocity(self._base_name) if self.cfg.robot == "unitree_h1" else None
        )
        student_obs, student_obs_dict = compute_student_observations(
            base_id=self._base_id,
            body_state=current_body_state,
            ref_motion_state=ref_motion_state,
            projected_gravity=base_gravity,
            last_actions=self.actions,
            history=self.history.entries,
            ref_episodic_offset=None,
            mask=self._mask,
            local_base_ang_velocity=local_base_ang_velocity,
        )
        obs_dict.update(student_obs_dict)
        obs_dict["student_policy"] = student_obs

        return obs_dict

    def _compute_extras(self) -> dict:
        """Compute the extras information used by evaluator.

        extras are required in the evaluator.collection() function, which expects
        ["data"]["body_pos_gt"], ["data"]["body_pos"] and ["termination_conditions"].

        Returns:
            dict: Extras, always an empty dict
        """
        extras = {}
        return extras

    def _get_episode_times(self):
        """Get the episode times in seconds."""
        return self.episode_length_buf * self.cfg.decimation * self.cfg.dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the done flags.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Should terminate flags and time out flags
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        ref_motion_state = self.reference_motion_manager.get_state_from_motion_lib_cache(
            episode_length_buf=self.episode_length_buf,
            offset=self._start_positions_on_terrain,
            terrain_heights=self.robot.get_terrain_heights(),
        )
        current_body_state = self._compose_body_state(
            extend_body_pos=self.extend_body_pos, extend_body_parent_ids=self.extend_body_parent_ids
        )

        # We don't yet use the correct contact forces in mujoco, therefore we just mock the values
        # to 0, s.t. this condition never triggers.
        # Dim: (num_envs, history_size, num_bodies, 3)
        net_contact_forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        undesired_contact_body_ids = [0]
        should_terminate, self._termination_conditions = check_termination_conditions(
            training_mode=self.cfg.mode.is_training_mode(),
            body_state=current_body_state,
            ref_motion_state=ref_motion_state,
            projected_gravity=self.robot.get_base_projected_gravity(self._base_name),
            gravity_x_threshold=self.cfg.gravity_x_threshold,
            gravity_y_threshold=self.cfg.gravity_y_threshold,
            ref_motion_mgr=self.reference_motion_manager,
            episode_times=self._get_episode_times(),
            max_ref_motion_dist=self.cfg.max_ref_motion_dist,
            in_recovery=None,
            net_contact_forces=net_contact_forces,
            undesired_contact_body_ids=undesired_contact_body_ids,
            mask=self._mask,
        )

        self.extras["termination_conditions"] = self._termination_conditions
        self.extras["time_outs"] = time_out

        return should_terminate, time_out
