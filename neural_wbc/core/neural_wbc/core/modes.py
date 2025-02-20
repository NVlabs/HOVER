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
import enum


class NeuralWBCModes(enum.Enum):
    TRAIN = 0
    TEST = 1
    DISTILL = 2
    DISTILL_TEST = 3

    def is_distill_mode(self):
        return self in {self.DISTILL, self.DISTILL_TEST}

    def is_training_mode(self):
        return self in {self.DISTILL, self.TRAIN}

    def is_test_mode(self):
        return self in {self.TEST, self.DISTILL_TEST}

    def is_distill_test_mode(self):
        return self == self.DISTILL_TEST
