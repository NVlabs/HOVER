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

import json
import numpy as np
import time
import torch
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm

from phc.smpllib.smpl_eval import compute_metrics_lite

from neural_wbc.core import EnvironmentWrapper


@dataclass
class Frame:
    """A frame contains a set of trajectory data."""

    body_pos: np.ndarray = np.array([])
    body_pos_masked: np.ndarray | None = None
    upper_body_joint_pos: np.ndarray = np.array([])
    lower_body_joint_pos: np.ndarray = np.array([])
    root_pos: np.ndarray = np.array([])
    root_lin_vel: np.ndarray = np.array([])
    root_rot: np.ndarray = np.array([])

    @staticmethod
    def from_dict(data: dict) -> Frame:
        """Create a Frame from a dictionary."""
        return Frame(**data)


@dataclass
class Episode:
    """An episode contains a set of trajectories, each is a sequence of frames."""

    body_pos: list[np.ndarray] = field(default_factory=list)
    body_pos_masked: list[np.ndarray] | None = None
    upper_body_joint_pos: list[np.ndarray] = field(default_factory=list)
    lower_body_joint_pos: list[np.ndarray] = field(default_factory=list)
    root_pos: list[np.ndarray] = field(default_factory=list)
    root_lin_vel: list[np.ndarray] = field(default_factory=list)
    root_rot: list[np.ndarray] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        """Number of frames in the episode."""
        return len(self.body_pos)

    def add_frame(self, frame: Frame):
        """Add a frame to the episode."""
        # Iterate through frame attributes and append them to episode
        for attr in vars(frame):
            # Handle body_pos_masked separately since it's optional
            if attr == "body_pos_masked":
                if frame.body_pos_masked is not None:
                    if self.body_pos_masked is None:
                        self.body_pos_masked = []
                    self.body_pos_masked.append(frame.body_pos_masked)
            # For all other attributes, append directly to corresponding episode list
            else:
                getattr(self, attr).append(getattr(frame, attr))

    def clear(self):
        """Clear internal buffers."""
        self.__init__()

    def filter(self, ids: list[int]) -> Episode:
        """Filter episode data to only include specified trajectory indices."""
        # Create new empty episode to store filtered data
        filtered = Episode()

        # Iterate through all attributes of this episode
        for attr, data in vars(self).items():
            # Only process attributes that have data
            if data is not None:
                # For each attribute, keep only the trajectories at the specified indices
                setattr(filtered, attr, [data[i] for i in ids])
        return filtered

    def trim(self, terminated_frame: torch.Tensor, end_id: int):
        """Helper method to cut data based on terminated frame."""
        trimmed = Episode()
        for attr, data in vars(self).items():
            if data is not None:
                setattr(trimmed, attr, self._trim_data(data, terminated_frame, end_id))
        return trimmed

    def _trim_data(self, data, terminated_frame, end_id):
        """Helper method to cut data based on terminated frame."""
        stacked = np.stack(data)
        return [stacked[:i, idx] for idx, i in enumerate(terminated_frame)][:end_id]


class MotionTrackingMetrics:
    """Class that aggregates motion tracking metrics throughout an evaluation."""

    def __init__(self):
        self.num_motions = 0
        self.success_rate = 0.0
        self.all_metrics = {}
        self.success_metrics = {}
        self.all_metrics_masked = {}
        self.success_metrics_masked = {}

        # Episodic data are stored as {"metric_name": {"means": [...], "weights": [...]}}
        # The final metrics are weighted sums of the means.
        self._all_metrics_by_episode = {}
        self._all_metrics_masked_by_episode = {}
        self._success_metrics_by_episode = {}
        self._success_metrics_masked_by_episode = {}

    def update(
        self,
        episode: Episode,
        episode_gt: Episode,
        success_ids: list,
    ):
        """Update and compute metrics for trajectories from all simulation instances in one episode."""
        self.num_motions += episode.num_frames
        # First, compute metrics on trajectories from all instances.
        self._compute_link_metrics(
            body_pos=episode.body_pos,
            body_pos_gt=episode_gt.body_pos,
            storage=self._all_metrics_by_episode,
        )
        self._compute_joint_metrics(
            episode=episode,
            episode_gt=episode_gt,
            storage=self._all_metrics_by_episode,
        )
        if episode.body_pos_masked and episode_gt.body_pos_masked:
            self._compute_link_metrics(
                body_pos=episode.body_pos_masked,
                body_pos_gt=episode_gt.body_pos_masked,
                storage=self._all_metrics_masked_by_episode,
            )

        if len(success_ids) == 0:
            return

        # Then, collect the trajectory from successful instances and compute metrics.
        success_episodes = episode.filter(success_ids)
        success_episodes_gt = episode_gt.filter(success_ids)

        self._compute_link_metrics(
            body_pos=success_episodes.body_pos,
            body_pos_gt=success_episodes_gt.body_pos,
            storage=self._success_metrics_by_episode,
        )
        self._compute_joint_metrics(
            episode=success_episodes,
            episode_gt=success_episodes_gt,
            storage=self._success_metrics_by_episode,
        )
        if success_episodes.body_pos_masked and success_episodes_gt.body_pos_masked:
            self._compute_link_metrics(
                body_pos=success_episodes.body_pos_masked,
                body_pos_gt=success_episodes_gt.body_pos_masked,
                storage=self._success_metrics_masked_by_episode,
            )

    def _compute_link_metrics(
        self,
        body_pos: list,
        body_pos_gt: list,
        storage: dict[str, dict[str, list[float]]],
    ):
        """Compute metrics of trajectories and save them by their means and number of elements (as weights)."""
        metrics = compute_metrics_lite(body_pos, body_pos_gt)
        self._store_metrics(metrics, storage)

    def _compute_joint_metrics(
        self,
        episode: Episode,
        episode_gt: Episode,
        storage: dict[str, dict[str, list[float]]],
    ):
        """Compute metrics of trajectories and save them by their means and number of elements (as weights)."""
        upper_body_joint_metrics = self._compute_joint_tracking_error(
            episode.upper_body_joint_pos, episode_gt.upper_body_joint_pos
        )
        lower_body_joint_metrics = self._compute_joint_tracking_error(
            episode.lower_body_joint_pos, episode_gt.lower_body_joint_pos
        )
        metrics = {
            "upper_body_joints_dist": upper_body_joint_metrics,
            "lower_body_joints_dist": lower_body_joint_metrics,
        }
        self._store_metrics(metrics, storage)

    def _compute_joint_tracking_error(self, joint_pos: list, joint_pos_gt: list):
        """Compute joint tracking error."""
        return np.array([np.mean(np.abs(joint_pos[i] - joint_pos_gt[i])) for i in range(len(joint_pos))])

    def _store_metrics(self, metrics: dict, storage: dict[str, dict[str, list[float]]]):
        """Store metrics by their means and number of elements (as weights)."""
        for key, value in metrics.items():
            if key not in storage:
                storage[key] = {"means": [], "weights": []}
            storage[key]["means"].append(np.mean(value).item())
            storage[key]["weights"].append(value.size)

    def conclude(self):
        """At the end of the evaluation, computes the metrics over all tasks."""
        self.all_metrics = {
            key: np.average(value["means"], weights=value["weights"])
            for key, value in self._all_metrics_by_episode.items()
        }
        self.success_metrics = {
            key: np.average(value["means"], weights=value["weights"])
            for key, value in self._success_metrics_by_episode.items()
        }
        self.all_metrics_masked = {
            key: np.average(value["means"], weights=value["weights"])
            for key, value in self._all_metrics_masked_by_episode.items()
        }
        self.success_metrics_masked = {
            key: np.average(value["means"], weights=value["weights"])
            for key, value in self._success_metrics_masked_by_episode.items()
        }

    def print(self):
        """Prints current metrics."""
        print(f"Number of reference motions: {self.num_motions}")
        print(f"Success Rate: {self.success_rate:.10f}")
        print("All: ", " \t\t".join([f"{k}: {v:.3f}" for k, v in self.all_metrics.items()]))
        print("Succ: ", " \t\t".join([f"{k}: {v:.3f}" for k, v in self.success_metrics.items()]))
        print("All Masked: ", " \t".join([f"{k}: {v:.3f}" for k, v in self.all_metrics_masked.items()]))
        print("Succ Masked: ", " \t".join([f"{k}: {v:.3f}" for k, v in self.success_metrics_masked.items()]))

    def save(self, directory: str):
        """Saves metrics to a time-stamped json file in ``directory``.

        Args:
            directory (str): Directory to stored the file to.
        """
        file_dir = Path(directory)
        file_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"{timestamp}.json"
        file_dir.joinpath(file_name)

        content = {
            "num_motions": self.num_motions,
            "success_rate": self.success_rate,
            "all": self.all_metrics,
            "success": self.success_metrics,
            "all_masked": self.all_metrics_masked,
            "success_masked": self.success_metrics_masked,
        }

        with open(file_dir.joinpath(file_name), "w") as fh:
            json.dump(content, fh)


class Evaluator:
    """A class for collecting data to evaluate motion tracking performance of an RL policy."""

    def __init__(
        self,
        env_wrapper: EnvironmentWrapper,
        metrics_path: str | None = None,
    ):
        """Initializes the evaluator.

        Args:
            env_wrapper (EnvironmentWrapper): The environment that the evaluation is taking place.
            metrics_path (str | None, optional): The directory that the metrics will be saved to. Defaults to None.
        """
        self._num_envs = env_wrapper.num_envs
        self._device = env_wrapper.device
        self._ref_motion_mgr = env_wrapper.reference_motion_manager

        self._ref_motion_start_id = 0
        self._num_unique_ref_motions = self._ref_motion_mgr.num_unique_motions
        self._ref_motion_frames = self._ref_motion_mgr.get_motion_num_steps()
        self._metrics = MotionTrackingMetrics()
        self._metrics_path = metrics_path

        # Episode data
        self._terminated = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._terminated_frame = self._ref_motion_frames.detach().clone()
        self._failed = torch.zeros((self._num_unique_ref_motions), dtype=torch.bool, device=self._device)
        self._episode = Episode()
        self._episode_gt = Episode()

        # Status
        self._pbar = tqdm(range(self._num_unique_ref_motions // self._num_envs), position=0, leave=True)
        self._curr_steps = 0
        self._num_episodes = 0

    def collect(self, dones: torch.Tensor, info: dict) -> bool:
        """Collects data from a step and updates internal states.

        Args:
            dones (torch.Tensor): environments that are terminated (failed) or truncated (timed out).
            info (dict): Extra information collected from a step.

        Returns:
            bool: Whether all current reference motions are evaluated and that all environments need a reset.
        """
        self._curr_steps += 1
        # Get the environments that terminated at the most recent step
        newly_terminated = torch.logical_and(~self._terminated, dones)
        self._collect_step_data(newly_terminated, info=info)

        self._terminated_frame[newly_terminated] = self._curr_steps
        self._terminated = torch.logical_or(self._terminated, dones)

        update_str = self._update_status_bar()

        if self._terminated.sum() == self._num_envs:
            self._aggregate_data()
            self._num_episodes += self._num_envs
            print(update_str)
            return True

        return False

    def _collect_step_data(self, newly_terminated: torch.Tensor, info: dict):
        """Collects data after each step.

        Args:
            newly_terminated(torch.Tensor(bool)): Newly terminated env
            info (dict): Extra information collected from a step.
        """

        state_data = info["data"]["state"]
        ground_truth_data = info["data"]["ground_truth"]

        body_pos = state_data["body_pos"]
        num_envs, num_bodies, _ = body_pos.shape

        mask = info["data"]["mask"]
        body_mask = mask[:, :num_bodies]
        body_mask = np.expand_dims(body_mask, axis=-1)
        body_mask_expanded = np.broadcast_to(body_mask, (num_envs, num_bodies, 3))

        upper_body_joint_ids = info["data"]["upper_joint_ids"]
        lower_body_joint_ids = info["data"]["lower_joint_ids"]

        frame = self._build_frame(state_data, body_mask_expanded, num_envs, upper_body_joint_ids, lower_body_joint_ids)
        frame_gt = self._build_frame(
            ground_truth_data, body_mask_expanded, num_envs, upper_body_joint_ids, lower_body_joint_ids
        )

        self._update_failure_metrics(newly_terminated, info)
        self._episode.add_frame(frame)
        self._episode_gt.add_frame(frame_gt)

    def _build_frame(
        self, data: dict, mask: np.ndarray, num_envs: int, upper_joint_ids: list, lower_joint_ids: list
    ) -> Frame:
        """Builds a frame from the data and mask.

        Args:
            data (dict): Dictionary containing trajectory data including body positions, joint positions, etc.
            mask (np.ndarray): Boolean mask array indicating which bodies to include in masked data.
            num_envs (int): Number of environments.
            upper_joint_ids (list): List of indices for upper body joints.
            lower_joint_ids (list): List of indices for lower body joints.

        Returns:
            Frame: A Frame object containing the processed trajectory data.
        """
        if np.any(mask):
            data["body_pos_masked"] = data["body_pos"][mask].reshape(num_envs, -1, 3)
        else:
            data["body_pos_masked"] = None

        joint_pos = data.pop("joint_pos")
        data["upper_body_joint_pos"] = joint_pos[:, upper_joint_ids]
        data["lower_body_joint_pos"] = joint_pos[:, lower_joint_ids]
        return Frame.from_dict(data)

    def _update_failure_metrics(self, newly_terminated: torch.Tensor, info: dict):
        """Updates failure metrics based on termination conditions."""
        start_id = self._ref_motion_start_id
        end_id = min(self._ref_motion_start_id + self._num_envs, self._num_unique_ref_motions)
        counted_envs = end_id - start_id

        # Get failure conditions excluding reference motion length
        failed_conditions = torch.stack(
            [
                v[:counted_envs].flatten()
                for k, v in info["termination_conditions"].items()
                if k != "reference_motion_length"
            ]
        )

        # Update failed environments
        self._failed[start_id:end_id] |= torch.logical_and(
            newly_terminated[:counted_envs], torch.any(failed_conditions, dim=0)
        )
        self._metrics.success_rate = 1 - torch.sum(self._failed).item() / end_id

    def _aggregate_data(self):
        """Aggregates data from one episode."""
        motion_end_id = min(self._num_envs, self._num_unique_ref_motions - self._num_episodes)
        self._episode = self._episode.trim(self._terminated_frame, motion_end_id)
        self._episode_gt = self._episode_gt.trim(self._terminated_frame, motion_end_id)

        # Get success IDs
        start_id = self._ref_motion_start_id
        end_id = min(start_id + self._num_envs, self._num_unique_ref_motions)
        success_ids = torch.nonzero(~self._failed[start_id:end_id]).flatten().tolist()

        # Update metrics
        self._metrics.update(
            episode=self._episode,
            episode_gt=self._episode_gt,
            success_ids=success_ids,
        )

    def _reset_data_buffer(self):
        """Resets data buffer for new episodes."""
        self._terminated[:] = False
        self._pbar.update(1)
        self._pbar.refresh()
        self._episode.clear()
        self._episode_gt.clear()
        self._ref_motion_frames = self._ref_motion_mgr.get_motion_num_steps()
        self._terminated_frame = self._ref_motion_frames.detach().clone()
        self._curr_steps = 0

    def _update_status_bar(self):
        """Updates status bar in the console to display current progress and selected metrics."""
        update_str = (
            f"Terminated: {self._terminated.sum().item()}  | max frames: {self._ref_motion_frames.max()} | steps"
            f" {self._curr_steps} | Start: {self._ref_motion_start_id} | Succ rate: {self._metrics.success_rate:.3f} |"
            f" Total failures: {self._failed.sum().item()} "
        )
        self._pbar.set_description(update_str)
        return update_str

    def is_evaluation_complete(self) -> bool:
        """Returns whether all reference motions in the dataset have been evaluated.

        Returns:
            bool: True if all reference motions have been evaluated.
        """
        return self._num_episodes >= self._num_unique_ref_motions

    def conclude(self):
        """Concludes evaluation by computing, printing and optionally saving metrics."""
        self._pbar.close()
        self._metrics.conclude()
        self._metrics.print()
        if self._metrics_path:
            self._metrics.save(self._metrics_path)

    def forward_motion_samples(self):
        """Steps forward in the list of reference motions.

        All simulated environments must be reset following this function call.
        """
        self._ref_motion_start_id += self._num_envs
        self._ref_motion_mgr.load_motions(random_sample=False, start_idx=self._ref_motion_start_id)
        self._reset_data_buffer()
