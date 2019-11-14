import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, sigma_type):
        super().__init__()
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.sigma = torch.tensor([np.sqrt(5)], dtype=torch.float32,
                                  device=self.train_device)  # TODO: Implement accordingly (T1, T2)
        self.sigma_type = sigma_type
        if sigma_type == "learn_sigma":
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
            sigma = self.sigma * np.sqrt(np.exp(-5 * 10**(-4) * ep))
        else:
            sigma = self.sigma  # TODO: Is it a good idea to leave it like this?

        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)
        return Normal(mu, sigma)
        # TODO: Add a layer for state value calculation (T3)


class Agent(object):
    def __init__(self, policy, norm, baseline   ):
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.baseline = baseline
        self.norm = norm

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
            .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []

        # TODO: Computact_log_probe discounted rewards (use the discount_rewards function)
        discounter_r = discount_rewards(rewards, self.gamma)
        if self.norm:
            discounter_r = (discounter_r - torch.mean(discounter_r)) / torch.std(discounter_r)
        # TODO: Compute critic loss and advantages (T3)
        # TODO: Compute the optimization term (T1, T3)
        gammas = torch.tensor([self.gamma ** t for t in range(len(rewards))]).to(self.train_device)
        update_term = -gammas * (discounter_r - self.baseline) * action_probs
        update = update_term.sum()
        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        update.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False, ep=None):
        x = torch.from_numpy(observation).float().to(self.train_device)
        # TODO: Pass state x through the policy network (T1)
        out_dist = self.policy(x, ep)
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

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
