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

import mujoco as mj

OBJECT_MAP = {
    "joint": mj.mjtObj.mjOBJ_JOINT,
    "body": mj.mjtObj.mjOBJ_BODY,
    "actuator": mj.mjtObj.mjOBJ_ACTUATOR,
}


def get_entity_name(model: mj.MjModel, entity_type: str, entity_id: int) -> str:
    """Gets name of an entity based on ID

    Args:
        model (mj.MjModel): model
        entity_type (str): entity type
        entity_id (int): entity id

    Returns:
        str: entity name
    """
    if entity_type == "body":
        return model.body(entity_id).name
    return mj.mj_id2name(model, OBJECT_MAP[entity_type], entity_id)


def get_entity_id(model: mj.MjModel, entity_type: str, entity_name: str) -> int:
    """Gets name of an entity based on ID

    Args:
        model (mj.MjModel): model
        entity_type (str): entity type
        entity_name (str): entity name

    Returns:
        int: entity id

    Notes:
        If the entity does not exist, returns -1
    """
    return mj.mj_name2id(model, OBJECT_MAP[entity_type], entity_name)


def has_free_joint(model: mj.MjModel) -> bool:
    """Check if the model has a free joint

    Returns:
        bool: True if the model has a free joint
    """
    return model.nv != model.nq
