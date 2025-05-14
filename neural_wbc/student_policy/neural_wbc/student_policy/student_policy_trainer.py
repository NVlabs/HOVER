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


import logging
import os
import statistics
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from neural_wbc.student_policy.policy import StudentPolicy
from neural_wbc.student_policy.storage import Slice, Storage
from neural_wbc.student_policy.student_policy_trainer_cfg import StudentPolicyTrainerCfg


class StudentPolicyTrainer:
    def __init__(self, env, cfg: StudentPolicyTrainerCfg):
        self._logger = logging.getLogger("student_policy_trainer")
        self._logger.setLevel(logging.INFO)

        self._env = env
        self._cfg = cfg
        self._num_envs = env.num_envs
        self._device = env.device
        self._storage = Storage(
            max_steps=self._cfg.num_steps_per_env,
            num_envs=self._env.num_envs,
            device=self._device,
            policy_obs_shape=[self._cfg.num_policy_obs],
            student_obs_shape=[self._cfg.num_student_obs],
            actions_shape=[self._cfg.num_actions],
        )

        self._teacher = self._cfg.teacher_policy
        self._iterations = 0
        self.start_iteration = 0  # tracking the loaded policy's iterations number
        self.start_time = time.time()
        self.tot_timesteps = 0
        self.writer = None
        self._initialize_student_network()

    def _initialize_student_network(self):
        self._student = StudentPolicy(
            self._cfg.num_student_obs,
            self._cfg.num_actions,
            self._cfg.policy_hidden_dims,
            self._cfg.activation,
            self._cfg.noise_std,
        ).to(self._device)
        self._optimizer = optim.Adam(self._student.parameters(), lr=self._cfg.learning_rate)

        if self._cfg.resume_path and not self._cfg.checkpoint:
            # Resume path is specified but not the checkpoint
            raise ValueError("A resume path is provided in the student trainer configuration without a checkpoint.")
        if not self._cfg.resume_path and self._cfg.checkpoint:
            # Checkpoint file provided without a resume path
            raise ValueError("A checkpoint is provided in the student trainer configuration without a resume path.")

        if self._cfg.resume_path and self._cfg.checkpoint:
            load_checkpoint_path = os.path.abspath(os.path.join(self._cfg.resume_path, self._cfg.checkpoint))
            self.load_student_network(load_checkpoint_path)

    def run(self):
        if self._cfg.student_policy_path is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self._cfg.student_policy_path, flush_secs=10)
            self._cfg.save(file_path=os.path.join(self._cfg.student_policy_path, "config.json"))

        obs_dict = self._env.get_full_observations()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        costbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self._env.num_envs, dtype=torch.float, device=self._device)
        cur_cost_sum = torch.zeros(self._env.num_envs, dtype=torch.float, device=self._device)
        cur_episode_length = torch.zeros(self._env.num_envs, dtype=torch.float, device=self._device)

        stop = time.time()
        while not self._should_stop():
            # Rollout:
            start = stop
            while not self._storage.is_full():
                with torch.inference_mode():
                    actions = self._produce_actions(obs_dict)
                    obs, privileged_obs, rewards, dones, infos = self._env.step(actions.detach())
                    gt_actions = self._get_ground_truth_actions(obs_dict["teacher"])
                    slice = Slice(
                        policy_observations=obs_dict["teacher"],
                        student_observations=obs_dict["student_policy"],
                        ground_truth_actions=gt_actions,
                        applied_actions=actions,
                    )
                    self._storage.add(slice=slice)
                    if self._cfg.student_policy_path is not None:
                        # Book keeping
                        if "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        if "cost" in infos:
                            cur_cost_sum += infos["cost"]
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        costbuffer.extend(cur_cost_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_cost_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                    obs_dict = infos["observations"]

            stop = time.time()
            collection_duration = stop - start

            # Learning step:
            start = stop
            mean_loss = self._update_student_network()
            stop = time.time()
            learn_duration = stop - start

            # Cleanup:
            start = stop
            self._storage.reset()
            if self._iterations % self._cfg.save_iteration == 0:
                self._save_student_network(
                    os.path.join(self._cfg.student_policy_path, f"model_{self._iterations + self.start_iteration}.pt")
                )

            self._iterations += 1
            stop = time.time()
            cleanup_duration = stop - start

            if self._cfg.student_policy_path is not None:
                self.log(locals())
            ep_infos.clear()

        self._save_student_network(os.path.join(self._cfg.student_policy_path, "final_model.pt"))

    def log(self, locs, width=80, pad=35):
        start = time.time()
        self.tot_timesteps += self._cfg.num_steps_per_env * self._env.num_envs
        total_duration = start - self.start_time
        iteration_duration = locs["collection_duration"] + locs["learn_duration"] + locs["cleanup_duration"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self._device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self._device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, self._iterations)
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self._student.std.mean()
        fps = int(self._cfg.num_steps_per_env * self._env.num_envs / iteration_duration)

        self.writer.add_scalar("Loss/learning_rate", self._cfg.learning_rate, self._iterations)
        self.writer.add_scalar("Loss/mean_loss", locs["mean_loss"], self._iterations)
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), self._iterations)

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), self._iterations)
            self.writer.add_scalar("Train/mean_cost", statistics.mean(locs["costbuffer"]), self._iterations)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), self._iterations)

        if "eval_info" in locs:
            # TODO: collect metrics during rollout and record them here
            self.writer.add_scalar("Eval/Success_rate", locs["eval_info"]["eval_success_rate"], self._iterations)

        str = (
            " \033[1m Learning iteration"
            f" {self._iterations + self.start_iteration} / {self._cfg.max_iteration + self.start_iteration} \033[0m "
        )

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_duration']:.3f}s, learning {locs['learn_duration']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean cost:':>{pad}} {statistics.mean(locs['costbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_duration']:.3f}s, learning {locs['learn_duration']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

        log_string += ep_string
        log_string += f"""{'-' * width}\n""" f"""{'Mean loss:':>{pad}} {locs['mean_loss']:.3f}\n"""

        eta = total_duration / (self._iterations + 1) * (self._cfg.max_iteration - self._iterations)
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_duration:.2f}s\n"""
            f"""{'Total time:':>{pad}} {total_duration:.2f}s\n"""
            f"""{'ETA:':>{pad}} {eta}s\n"""
        )

        log_string += f"""path: {self._cfg.student_policy_path}\n"""
        print("\r " + log_string, end="")

        # Log perf last to delay stop as much as possible.
        self.writer.add_scalar("Perf/total_fps", fps, self._iterations)
        self.writer.add_scalar("Perf/collection_duration", locs["collection_duration"], self._iterations)
        self.writer.add_scalar("Perf/learn_duration", locs["learn_duration"], self._iterations)
        self.writer.add_scalar("Perf/cleanup_duration", locs["cleanup_duration"], self._iterations)
        self.writer.add_scalar("Perf/total_duration", total_duration, self._iterations)
        stop = time.time()
        log_duration = stop - start
        self.writer.add_scalar("Perf/log_duration", log_duration, self._iterations)

    def _should_stop(self):
        reached_max_iteration = self._iterations >= self._cfg.max_iteration
        return reached_max_iteration or self._has_converged()

    def _has_converged(self) -> bool:
        # TODO: do the actual math, P1 task
        return False

    def _produce_actions(self, obs: dict) -> torch.Tensor:
        """
        Roll out the environment with either the expert policy or the student policy, depending on
        the current state of training.
        """
        if self._iterations + self.start_iteration >= self._cfg.student_rollout_iteration:
            observations = obs["student_policy"]
            action = self._student.act(observations)
        else:
            action = self._teacher.act_rollout(obs["teacher"])

        return action

    def _get_ground_truth_actions(self, policy_obs: torch.Tensor) -> torch.Tensor:
        return self._teacher.act(policy_obs)

    def _update_student_network(self) -> float:
        mean_loss = 0
        generator = self._storage.mini_batch_generator(self._cfg.num_mini_batches, self._cfg.num_learning_epochs)

        for (
            policy_observations_batch,
            student_observations_batch,
            ground_truth_actions_batch,
            applied_actions_batch,
        ) in generator:
            pred_action = self._student.act_inference(student_observations_batch)
            kin_loss = torch.norm(pred_action - ground_truth_actions_batch, dim=-1).mean()  # RMSE
            kin_loss = kin_loss * self._cfg.dagger_coefficient
            self._optimizer.zero_grad()
            kin_loss.backward()
            nn.utils.clip_grad_norm_(self._student.parameters(), self._cfg.max_grad_norm)
            self._optimizer.step()
            mean_loss += kin_loss.item()

        num_updates = self._cfg.num_learning_epochs * self._cfg.num_mini_batches
        mean_loss = mean_loss / num_updates
        return mean_loss

    def _save_student_network(self, path):
        self._logger.info(f"Saving student policy to {path}")
        torch.save(
            {
                "model_state_dict": self._student.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "iter": self.start_iteration + self._iterations,
            },
            path,
        )

    def load_student_network(self, path, load_optimizer=True):
        print("Loading :", path)
        loaded_dict = torch.load(path, map_location=self._device)
        self._student.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self._optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.start_iteration = loaded_dict["iter"]
        return self._student

    def get_inference_policy(self, dtype=torch.float32, device=None):
        # for playing
        self._student.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self._student.to(dtype=dtype, device=device)
        return self._student.act_inference

    def get_student(self, device=None):
        # Returns the network, e.g. to translate network between libraries or programming languages
        if device is not None:
            self._student.to(device)
        return self._student
