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


import torch
import torch.nn as nn
from torch.distributions import Normal


class StudentPolicy(nn.Module):
    def __init__(self, num_obs, num_actions, policy_hidden_dims=[256, 256, 256], activation="elu", noise_std=0.001):
        super().__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_obs

        # Policy
        policy_layers = []
        policy_layers.append(nn.Linear(mlp_input_dim_a, policy_hidden_dims[0]))
        policy_layers.append(activation)
        for layer in range(len(policy_hidden_dims)):
            if layer == len(policy_hidden_dims) - 1:
                policy_layers.append(nn.Linear(policy_hidden_dims[layer], num_actions))
            else:
                policy_layers.append(nn.Linear(policy_hidden_dims[layer], policy_hidden_dims[layer + 1]))
                policy_layers.append(activation)
        self.policy = nn.Sequential(*policy_layers)

        print(f"Student Policy MLP: {self.policy}")

        # Action noise
        self.std = nn.Parameter(noise_std * torch.ones(num_actions))
        self.std.requires_grad = False
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.policy(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.policy(observations)
        return actions_mean

    def load(self, path, device):
        self.to(device)
        loaded_dict = torch.load(path, map_location=device)
        self.load_state_dict(loaded_dict["model_state_dict"])


def get_activation(act_name):
    activation_functions = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }

    return activation_functions.get(act_name, lambda: print(f"invalid activation function {act_name}!"))()
