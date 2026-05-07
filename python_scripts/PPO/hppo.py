import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from python_scripts.Project_config import device


class MultiDiscreteActorCritic(nn.Module):
    def __init__(self, num_servos, node_num, state_dim=20, use_image_input=True):
        super().__init__()
        self.num_servos = num_servos
        self.node_num = node_num
        self.state_dim = int(state_dim)
        self.use_image_input = bool(use_image_input)

        self.conv1 = nn.Conv2d(1, 32, (5, 5), stride=(2, 2), padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, (5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(32, 32, (5, 5), stride=(2, 2), padding=1)
        self.fc0 = nn.Linear(6272, 6000)
        self.fc1 = nn.Linear(6000, 100)
        self.fc2 = nn.Linear(self.state_dim, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc_graph = nn.Linear(self.state_dim, 100)
        self.fc4 = nn.Linear(300, 200)
        self.discrete_head = nn.Linear(200, num_servos)
        self.critic = nn.Linear(200, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @staticmethod
    def _min_max_normalize(tensor):
        value_range = torch.max(tensor) - torch.min(tensor)
        if value_range < 1e-8:
            return torch.zeros_like(tensor)
        return (tensor - torch.min(tensor)) / (value_range + 1e-8)

    def _extract_image_features(self, x):
        if not self.use_image_input or x is None:
            return torch.zeros(100, dtype=torch.float32, device=device)

        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        if torch.isnan(x).any():
            print("Warning: image input contains NaN; replacing with 0.")
            x = torch.nan_to_num(x, nan=0.0)

        x = torch.unsqueeze(x, dim=0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = torch.flatten(x)
        x = self.fc1(self.fc0(x))
        return self._min_max_normalize(x)

    def forward(self, x, state, x_graph):
        image_features = self._extract_image_features(x)

        state = torch.as_tensor(state, dtype=torch.float32).to(device)
        if torch.isnan(state).any():
            print("Warning: state input contains NaN; replacing with 0.")
            state = torch.nan_to_num(state, nan=0.0)
        state_features = self.fc3(self.fc2(state))
        state_features = self._min_max_normalize(state_features)

        x_graph = torch.as_tensor(x_graph, dtype=torch.float32).to(device)
        if torch.isnan(x_graph).any():
            print("Warning: graph input contains NaN; replacing with 0.")
            x_graph = torch.nan_to_num(x_graph, nan=0.0)
        graph_features = self.fc_graph(x_graph)
        graph_features = self._min_max_normalize(graph_features)

        merged = torch.cat((image_features, state_features, graph_features), dim=-1)
        features = self.fc4(merged)

        discrete_logits = self.discrete_head(features)
        discrete_probs = torch.sigmoid(discrete_logits)
        if torch.isnan(discrete_probs).any():
            print("Warning: discrete_probs contains NaN; forcing to 0.5.")
            discrete_probs = torch.full_like(discrete_probs, 0.5)

        value = self.critic(features)
        return discrete_probs, value


class HPPO:
    def __init__(self, num_servos, node_num, env_information=None, state_dim=20, use_image_input=True):
        self.num_servos = num_servos
        self.node_num = node_num
        self.env_information = env_information
        self.state_dim = int(state_dim)
        self.use_image_input = bool(use_image_input)

        self.gamma = 0.99
        self.gae_lambda = 0.90
        self.clip_ratio = 0.15
        self.policy_update_epochs = 2
        self.value_coef = 0.25
        self.entropy_coef = 0.01
        self.lr = 3e-5
        self.lr_decay = 0.995

        self.policy = MultiDiscreteActorCritic(
            num_servos,
            node_num,
            state_dim=self.state_dim,
            use_image_input=self.use_image_input,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.log_probs = []
        self.dones = []

    @staticmethod
    def _looks_like_observation_tuple(obs):
        if isinstance(obs, tuple) and len(obs) >= 2:
            return True
        if isinstance(obs, list) and len(obs) in (2, 3):
            first = obs[0]
            return not np.isscalar(first)
        return False

    @classmethod
    def _split_obs(cls, obs, x_graph=None):
        if cls._looks_like_observation_tuple(obs):
            img_input = obs[0]
            state_input = obs[1]
            graph_input = x_graph if x_graph is not None else (obs[2] if len(obs) > 2 else state_input)
        else:
            img_input = None
            state_input = obs
            graph_input = x_graph if x_graph is not None else obs
        return img_input, state_input, graph_input

    def choose_action(self, obs, x_graph=None):
        with torch.no_grad():
            img_input, state_input, graph_input = self._split_obs(obs, x_graph)
            discrete_probs, value = self.policy(x=img_input, state=state_input, x_graph=graph_input)
            distribution = Bernoulli(discrete_probs)
            discrete_actions = distribution.sample()
            discrete_log_probs = distribution.log_prob(discrete_actions)
            return {
                "discrete_action": discrete_actions.cpu().numpy(),
                "discrete_log_prob": discrete_log_probs.cpu().numpy(),
                "value": value.item(),
            }

    def store_transition(self, state, action, reward, next_state, done, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def calculate_advantages(self):
        advantages = []
        gae = 0
        max_gae = 10.0

        for i in reversed(range(len(self.rewards))):
            next_value = 0 if i == len(self.rewards) - 1 else self.values[i + 1]
            delta = self.rewards[i] + self.gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[i]) * gae
            gae = np.clip(gae, -max_gae, max_gae)
            advantages.insert(0, gae)

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
        return (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

    def _clear_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def learn(self):
        if len(self.states) < 32:
            return 0

        advantages = self.calculate_advantages()
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            print("Warning: advantage contains NaN or Inf; skipping decision learn.")
            self._clear_buffer()
            return 0

        if torch.abs(advantages).max().item() > 50:
            advantages = torch.clamp(advantages, -10, 10)

        returns = advantages + torch.tensor(self.values, dtype=torch.float32).to(device)
        batch_states = self.states
        batch_discrete_actions = torch.tensor(self.actions, dtype=torch.float32).to(device)
        batch_discrete_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for _ in range(self.policy_update_epochs):
            all_discrete_probs = []
            all_values = []
            for state in batch_states:
                img_input, state_input, graph_input = self._split_obs(state)
                discrete_probs, value = self.policy(x=img_input, state=state_input, x_graph=graph_input)
                all_discrete_probs.append(discrete_probs)
                all_values.append(value)

            all_discrete_probs = torch.stack(all_discrete_probs)
            all_values = torch.cat(all_values)

            distribution = Bernoulli(all_discrete_probs)
            new_discrete_log_probs = distribution.log_prob(batch_discrete_actions)
            discrete_ratio = torch.exp(new_discrete_log_probs - batch_discrete_log_probs)
            total_ratio = discrete_ratio.mean(dim=1)

            surr1 = total_ratio * advantages
            surr2 = torch.clamp(total_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(all_values, returns)
            entropy = distribution.entropy().mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss.item()

        self.scheduler.step()
        self._clear_buffer()
        return total_loss / self.policy_update_epochs
