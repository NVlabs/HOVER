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
import copy
import numpy as np
import torch
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from neural_wbc.core import EnvironmentWrapper
from neural_wbc.core.evaluator import Episode, Evaluator, MotionTrackingMetrics
from neural_wbc.core.reference_motion import ReferenceMotionManager, ReferenceMotionManagerCfg
from neural_wbc.data import get_data_path


class TestMotionTrackingMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = MotionTrackingMetrics()

    def test_initial_state(self):
        self.assertEqual(self.metrics.success_rate, 0.0)
        self.assertEqual(self.metrics.num_motions, 0)
        self.assertEqual(self.metrics.all_metrics, {})
        self.assertEqual(self.metrics.success_metrics, {})

    @patch("neural_wbc.core.evaluator.compute_metrics_lite")  # Mock the compute_metrics_lite function
    def test_update(self, mock_compute_metrics_lite):
        mock_compute_metrics_lite.return_value = {"metric1": np.array([1, 2, 3]), "metric2": np.array([4, 5, 6])}
        episode = Episode()
        episode_gt = Episode()

        episode.body_pos = [np.array([1, 2]), np.array([3, 4])]
        episode_gt.body_pos = [np.array([1, 2]), np.array([3, 4])]
        episode.upper_body_joint_pos = [np.array([1, 2]), np.array([3, 4])]
        episode_gt.upper_body_joint_pos = [np.array([1, 2]), np.array([3, 4])]
        episode.lower_body_joint_pos = [np.array([1, 2]), np.array([3, 4])]
        episode_gt.lower_body_joint_pos = [np.array([1, 2]), np.array([3, 4])]
        episode.root_pos = [np.array([1, 2]), np.array([3, 4])]
        episode_gt.root_pos = [np.array([1, 2]), np.array([3, 4])]
        episode.root_lin_vel = [np.array([1, 2]), np.array([3, 4])]
        episode_gt.root_lin_vel = [np.array([1, 2]), np.array([3, 4])]
        episode.root_rot = [np.array([1, 2]), np.array([3, 4])]
        episode_gt.root_rot = [np.array([1, 2]), np.array([3, 4])]

        success_ids = [0]

        self.metrics.update(
            episode=episode,
            episode_gt=episode_gt,
            success_ids=success_ids,
        )

        self.assertEqual(self.metrics.num_motions, 2)
        self.assertIn("metric1", self.metrics._all_metrics_by_episode)
        self.assertIn("metric1", self.metrics._success_metrics_by_episode)

    def test_conclude(self):
        self.metrics._all_metrics_by_episode = {"metric1": {"means": [1, 2], "weights": [1, 1]}}
        self.metrics._success_metrics_by_episode = {"metric1": {"means": [1, 2], "weights": [1, 1]}}

        self.metrics.conclude()

        self.assertEqual(self.metrics.all_metrics["metric1"], 1.5)
        self.assertEqual(self.metrics.success_metrics["metric1"], 1.5)

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("json.dump")
    @patch("time.strftime", return_value="20240101-000000")
    def test_save(self, mock_strftime, mock_json_dump, mock_open):
        self.metrics.save("/tmp")
        mock_open.assert_called_once_with(Path("/tmp/20240101-000000.json"), "w")
        mock_json_dump.assert_called_once()


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        env_wrapper = Mock(spec=EnvironmentWrapper)
        env_wrapper.num_envs = 2
        env_wrapper.device = "cpu"

        env_wrapper.reference_motion_manager = Mock()
        env_wrapper.reference_motion_manager.num_unique_motions = 10
        env_wrapper.reference_motion_manager.get_motion_num_steps.return_value = torch.tensor([100, 100])
        self.evaluator = Evaluator(env_wrapper)

    def test_initial_state(self):
        self.assertEqual(self.evaluator._num_envs, 2)
        self.assertEqual(self.evaluator._device, "cpu")
        self.assertEqual(self.evaluator._num_unique_ref_motions, 10)
        self.assertEqual(len(self.evaluator._terminated), 2)
        self.assertEqual(len(self.evaluator._failed), 10)

    def test_collect(self):
        dones = torch.tensor([True, False])
        info = {
            "data": {"body_pos_gt": np.array([[1, 2], [3, 4]]), "body_pos": np.array([[1, 2], [3, 4]])},
            "termination_conditions": {"reference_motion_length": torch.tensor([False, False])},
        }
        with patch.object(self.evaluator, "_collect_step_data"):
            result = self.evaluator.collect(dones, info)
            self.assertFalse(result)

    def test_aggregate_data(self):
        self.evaluator._episode_gt.body_pos = [np.array([[1, 2], [3, 4]])]
        self.evaluator._episode.body_pos = [np.array([[1, 2], [3, 4]])]
        self.evaluator._episode_gt.body_pos_masked = [np.array([[1, 2], [3, 4]])]
        self.evaluator._episode.body_pos_masked = [np.array([[1, 2], [3, 4]])]
        self.evaluator._episode_gt.upper_body_joint_pos = [np.array([[1, 2], [3, 4]])]
        self.evaluator._episode.upper_body_joint_pos = [np.array([[1, 2], [3, 4]])]
        self.evaluator._episode_gt.lower_body_joint_pos = [np.array([[1, 2], [3, 4]])]
        self.evaluator._episode.lower_body_joint_pos = [np.array([[1, 2], [3, 4]])]
        self.evaluator._episode_gt.root_pos = [np.array([[1, 2], [3, 4]])]
        self.evaluator._episode.root_pos = [np.array([[1, 2], [3, 4]])]
        self.evaluator._episode_gt.root_lin_vel = [np.array([[1, 2], [3, 4]])]
        self.evaluator._episode.root_lin_vel = [np.array([[1, 2], [3, 4]])]
        self.evaluator._episode_gt.root_rot = [np.array([[1, 2], [3, 4]])]
        self.evaluator._episode.root_rot = [np.array([[1, 2], [3, 4]])]

        self.evaluator._terminated_frame = torch.tensor([100, 100])
        self.evaluator._terminated = torch.tensor([True, True])
        self.evaluator._ref_motion_start_id = 0
        self.evaluator._num_episodes = 0

        with patch.object(self.evaluator._metrics, "update") as mock_update:
            self.evaluator._aggregate_data()
            mock_update.assert_called_once()

    def test_conclude(self):
        with patch.object(self.evaluator._metrics, "conclude") as mock_conclude, patch.object(
            self.evaluator._metrics, "print"
        ) as mock_print, patch("neural_wbc.core.evaluator.tqdm.close"):
            self.evaluator.conclude()
            mock_conclude.assert_called_once()
            mock_print.assert_called_once()


class TestEvaluatorWithReferenceManager(unittest.TestCase):
    num_envs = 1
    device = torch.device("cpu")

    def setUp(self):
        env_wrapper = Mock(spec=EnvironmentWrapper)
        env_wrapper.num_envs = self.num_envs
        env_wrapper.device = self.device

        env_wrapper.reference_motion_manager = self._create_reference_motion_manager()
        self.evaluator = Evaluator(env_wrapper)

    def _create_reference_motion_manager(self):
        cfg = ReferenceMotionManagerCfg()
        cfg.motion_path = get_data_path("motions/stable_punch.pkl")
        cfg.skeleton_path = get_data_path("motion_lib/h1.xml")

        return ReferenceMotionManager(
            cfg=cfg,
            device=self.device,
            num_envs=self.num_envs,
            random_sample=False,
            extend_head=False,
            dt=0.005,
        )

    def _create_dones_and_info(self, ref_motion_state):
        # Construct a test data that fails at first step.
        dones = torch.tensor([True])
        info = {
            "data": {
                "state": {},
                "ground_truth": {},
                "upper_joint_ids": range(0, 10),
                "lower_joint_ids": range(11, 19),
            },
            "termination_conditions": {"gravity": torch.tensor([[True]])},
        }
        # Only create mask for the bodies as that's the only thing this test cares for.
        num_bodies = ref_motion_state.body_pos_extend.shape[1]
        info["data"]["mask"] = torch.zeros((self.num_envs, num_bodies), dtype=torch.bool)
        info["data"]["mask"][:, -3:] = True

        # Convert reference motion state to numpy arrays and copy to both ground truth and state
        for key in ["body_pos_extend", "joint_pos", "root_pos", "root_lin_vel", "root_rot"]:
            numpy_data = getattr(ref_motion_state, key).cpu().detach().numpy()
            info_key = key.replace("_extend", "")  # handle body_pos_extend special case
            info["data"]["ground_truth"][info_key] = numpy_data
            info["data"]["state"][info_key] = numpy_data
        return dones, info

    def test_forward_motion_without_offset(self):
        # The internal motion manager cache will always reset if the offset is None.
        # Load the initial frame of the first motion
        episode_length_buf = torch.Tensor([0] * self.num_envs, device=self.device).long()
        ref_motion_state1 = self.evaluator._ref_motion_mgr.get_state_from_motion_lib_cache(
            episode_length_buf=episode_length_buf, offset=None
        )

        dones, info = self._create_dones_and_info(ref_motion_state=ref_motion_state1)
        reset_env = self.evaluator.collect(dones=dones, info=info)
        # Confirm reset is true and
        self.assertTrue(reset_env)
        self.assertFalse(self.evaluator.is_evaluation_complete())

        # Forward the motion lib.
        self.evaluator.forward_motion_samples()
        ref_motion_state2 = self.evaluator._ref_motion_mgr.get_state_from_motion_lib_cache(
            episode_length_buf=episode_length_buf,
            offset=None,
        )

        # Confirm that the motion is updated properly
        self.assertFalse(torch.equal(ref_motion_state1.joint_pos, ref_motion_state2.joint_pos))

    def test_forward_motion_with_offset(self):
        # Load the initial frame of the first motion
        episode_length_buf = torch.Tensor([0] * self.num_envs, device=self.device).long()
        start_positions_on_terrain = torch.zeros([self.num_envs, 3], device=self.device)
        ref_motion_state1 = self.evaluator._ref_motion_mgr.get_state_from_motion_lib_cache(
            episode_length_buf=episode_length_buf, offset=start_positions_on_terrain
        )

        dones, info = self._create_dones_and_info(ref_motion_state=ref_motion_state1)
        reset_env = self.evaluator.collect(dones=dones, info=info)
        # Confirm reset is true and the evaluation is not complete.
        self.assertTrue(reset_env)
        self.assertFalse(self.evaluator.is_evaluation_complete())

        # Forward the motion lib.
        self.evaluator.forward_motion_samples()
        ref_motion_state2 = self.evaluator._ref_motion_mgr.get_state_from_motion_lib_cache(
            episode_length_buf=episode_length_buf, offset=start_positions_on_terrain
        )

        # Confirm that the motion is updated. When we forward motion samples, we move on to the next motion.
        self.assertFalse(torch.equal(ref_motion_state1.joint_pos, ref_motion_state2.joint_pos))

        # Set the episode length properly with make the motion cache updated properly.
        episode_length_buf += 1
        ref_motion_state3 = self.evaluator._ref_motion_mgr.get_state_from_motion_lib_cache(
            episode_length_buf=episode_length_buf, offset=start_positions_on_terrain
        )
        self.assertFalse(torch.equal(ref_motion_state2.joint_pos, ref_motion_state3.joint_pos))


class TestMaskedMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = MotionTrackingMetrics()

    def _create_data(
        self,
    ) -> tuple[
        list[bool],
        Episode,
        Episode,
    ]:
        success_ids = [True]

        episode = Episode()
        episode_gt = Episode()

        episode_gt.body_pos = [
            np.array(
                [
                    [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
                    [[2, 0, 0], [3, 0, 0], [4, 0, 0]],
                    [[3, 0, 0], [4, 0, 0], [5, 0, 0]],
                ],
                dtype=np.float64,
            ),
            np.array(
                [
                    [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
                    [[2, 0, 0], [3, 0, 0], [4, 0, 0]],
                    [[3, 0, 0], [4, 0, 0], [5, 0, 0]],
                ],
                dtype=np.float64,
            ),
        ]

        episode_gt.upper_body_joint_pos = [np.random.rand(3, 3)] * 2
        episode_gt.lower_body_joint_pos = [np.random.rand(3, 3)] * 2
        episode_gt.root_pos = [np.random.rand(3, 1)] * 2
        episode_gt.root_lin_vel = [np.random.rand(3, 1)] * 2
        episode_gt.root_rot = [np.random.rand(3, 1)] * 2

        episode.body_pos = copy.deepcopy(episode_gt.body_pos.copy())
        episode.upper_body_joint_pos = copy.deepcopy(episode_gt.upper_body_joint_pos)
        episode.lower_body_joint_pos = copy.deepcopy(episode_gt.lower_body_joint_pos)
        episode.root_pos = copy.deepcopy(episode_gt.root_pos)
        episode.root_lin_vel = copy.deepcopy(episode_gt.root_lin_vel)
        episode.root_rot = copy.deepcopy(episode_gt.root_rot)

        mask = np.array([False, True, True])
        episode_gt.body_pos_masked = [pos[:, mask, :] for pos in episode_gt.body_pos.copy()]
        episode.body_pos_masked = [pos[:, mask, :] for pos in episode.body_pos.copy()]

        return (
            success_ids,
            episode,
            episode_gt,
        )

    def test_zero_metrics(self):
        """
        Test the metrics when all bodies have zero error.
        """

        (
            success_ids,
            episode,
            episode_gt,
        ) = self._create_data()

        self.metrics.update(
            success_ids=success_ids,
            episode=episode,
            episode_gt=episode_gt,
        )

        self.metrics.conclude()

        for name, value in self.metrics.all_metrics.items():
            self.assertAlmostEqual(value, 0.0)

        for name, value in self.metrics.success_metrics.items():
            self.assertAlmostEqual(value, 0.0)

        for name, value in self.metrics.all_metrics_masked.items():
            self.assertAlmostEqual(value, 0.0)

        for name, value in self.metrics.success_metrics_masked.items():
            self.assertAlmostEqual(value, 0.0)

    def test_offset_metrics(self):
        """
        Test the metrics when all bodies have only a translation error to GT.
        Then we expect the global error to b the translation, the local error should be 0.
        """

        (
            success_ids,
            episode,
            episode_gt,
        ) = self._create_data()

        offset = np.array([0.1, 0.0, 0.0])
        for i in range(episode.num_frames):
            episode.body_pos[i] += offset
            episode.body_pos_masked[i] += offset

        self.metrics.update(
            success_ids=success_ids,
            episode=episode,
            episode_gt=episode_gt,
        )

        self.metrics.conclude()

        self.assertAlmostEqual(self.metrics.all_metrics["mpjpe_g"], np.linalg.norm(offset) * 1000)
        self.assertAlmostEqual(self.metrics.all_metrics["mpjpe_l"], 0.0)

        self.assertAlmostEqual(self.metrics.all_metrics_masked["mpjpe_g"], np.linalg.norm(offset) * 1000)
        self.assertAlmostEqual(self.metrics.all_metrics_masked["mpjpe_l"], 0.0)

    def test_masked_metrics(self):
        """
        Test the metrics when only the masked bodies have an offset. We expect the masked error to
        be greater than the unmasked error.
        """
        (
            success_ids,
            episode,
            episode_gt,
        ) = self._create_data()

        offset = np.array([0.1, 0.0, 0.0])
        for i in range(episode.num_frames):
            episode.body_pos[i][2, :] += offset
            episode.body_pos_masked[i] += offset

        self.metrics.update(
            success_ids=success_ids,
            episode=episode,
            episode_gt=episode_gt,
        )

        self.metrics.conclude()

        self.assertAlmostEqual(self.metrics.all_metrics["mpjpe_g"], np.linalg.norm(offset) * 3 / 9 * 1000)
        self.assertAlmostEqual(self.metrics.all_metrics["mpjpe_l"], 0.0)

        self.assertAlmostEqual(self.metrics.all_metrics_masked["mpjpe_g"], np.linalg.norm(offset) * 1000)
        self.assertAlmostEqual(self.metrics.all_metrics_masked["mpjpe_l"], 0.0)


if __name__ == "__main__":
    unittest.main()
