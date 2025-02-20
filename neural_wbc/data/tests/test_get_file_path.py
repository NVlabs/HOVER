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


import os
import pathlib
import unittest

from neural_wbc.data import get_data_path


class TestGetDataPath(unittest.TestCase):
    def setUp(self):
        self.data_dir = pathlib.Path(get_data_path("")).resolve()
        self.existing_files = [str(f) for f in self.data_dir.glob("**/*") if f.is_file()]
        self.non_existing_file = "non_existing_file.txt"

    def test_existing_files(self):
        self.assertTrue(str(self.data_dir).endswith("neural_wbc/data/data"))

        for rel_path in self.existing_files:
            with self.subTest(rel_path=rel_path):
                data_path = get_data_path(rel_path)
                self.assertTrue(os.path.isfile(data_path), f"File should exist: {data_path}")

    def test_non_existing_file(self):
        with self.assertRaises(FileNotFoundError):
            get_data_path(self.non_existing_file)


if __name__ == "__main__":
    unittest.main()
