import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards


class Value(torch.nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_space = state_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, sigma_type):
        super().__init__()
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.sigma = torch.tensor([5], dtype=torch.float32,
                                  device=self.train_device)  # TODO: Implement accordingly (T1, T2)
        self.sigma_type = sigma_type
        if str(sigma_type) == "learn_sigma":
            self.sigma = torch.nn.Parameter(self.sigma)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, ep=None):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc2_mean(x)
        if self.sigma_type == "exp_decay":
            sigma = self.sigma * np.exp(-5 * 10 ** (-4) * ep)
        else:
            sigma = self.sigma  # TODO: Is it a good idea to leave it like this?

        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)
        return Normal(mu, sigma)
        # TODO: Add a layer for state value calculation (T3)


class Agent(object):
    def __init__(self, policy, norm, baseline, value_fn):
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.policy = policy.to(self.train_device)
        self.value_fn = value_fn.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(list(policy.parameters()) + list(value_fn.parameters()), lr=4e-3)

        self.gamma = 0.98
        self.states = []
        self.values = []
        self.action_probs = []
        self.rewards = []
        self.baseline = baseline
        self.norm = norm
        self.done = []

    def episode_finished(self):
        action_probs = torch.stack(self.action_probs, dim=0) \
            .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = self.done
        if val is None:
            val=0
        discounter_r = discount_rewards(rewards, self.gamma, done, recurse=True, running_add=val)
        values = torch.stack(self.values, 0).to(self.train_device).squeeze(-1)
        if self.norm:
            discounter_r = (discounter_r - torch.mean(discounter_r)) / torch.std(discounter_r)
        self.states, self.action_probs, self.rewards, self.values, self.done = [], [], [], [], []
        # TODO: Compute critic loss and advantages (T3)
        delta = discounter_r - values
        policy_update = torch.mean(-delta.detach() * action_probs)
        val_fn_update = torch.mean(torch.pow(delta, 2))
        # TODO: Compute the optimization term (T1, T3)
        total_update = val_fn_update + policy_update
        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        total_update.backward()
        torch.nn.utils.clip_grad_norm(list(self.policy.parameters())+list(self.value_fn.parameters()), max_norm=0.8)
        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)
        # TODO: Pass state x through the policy network (T1)
        out_dist = self.policy(x)
        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            return out_dist.mean
        else:
            action = out_dist.sample()
        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = out_dist.log_prob(action)
        # TODO: Return state value prediction, and/or save it somewhere (T3)

        return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward, value, done=None):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.values.append(value)
        self.done.append(done)
