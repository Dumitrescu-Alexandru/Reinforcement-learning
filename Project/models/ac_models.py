import numpy as np
from collections import namedtuple
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from utils import Transition, ReplayMemory, discount_rewards
import torchvision.models as models
from time import time


class FeatExtractConv(nn.Module):
    def __init__(self, train_device=torch.device("cuda:0"), channels=3):
        super(FeatExtractConv, self).__init__()
        self.train_device = train_device
        self.conv1 = nn.Conv2d(channels, 4, kernel_size=(7, 7), stride=1)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=(5, 5), stride=1)
        # self.max_pool1 = nn.MaxPool2d(2, stride=2)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 4), stride=2)
        # self.max_pool2 = nn.MaxPool2d(2, stride=2)
        # self.conv3 = nn.Conv2d(3, 3, kernel_size=(5, 5), stride=2)
        # self.max_pool3 = nn.MaxPool2d(2, stride=2)
        self.train_device = train_device

    def forward(self, x):
        x = x.to(self.train_device)
        x = F.relu6(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu6(self.conv2(x))
        # x = F.relu6(self.max_pool1(self.conv1(x)))
        # x = F.relu6(self.max_pool2(self.conv2(x)))
        # x = F.relu6(self.max_pool3(self.conv3(x)))
        return x


class Policy(nn.Module):
    def __init__(self, hidden=100, fine_tune=True, train_device=torch.device("cuda:0"),
                 history=3, down_sample=False, gray_scale=False):
        super(Policy, self).__init__()
        self.train_device = train_device
        self.hidden = hidden
        self.history = history
        self.channels = 1 if gray_scale else 3
        self.down_factor = 4 if down_sample else 1
        self.feature_extractor = FeatExtractConv(channels=self.channels)
        # self.feature_extractor = nn.Linear(3 * 2500, hidden)
        if not fine_tune:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        self.hidden_layer = nn.Linear(1 * 68 * 18, hidden)
        self.output = nn.Linear(hidden, 3)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        x = torch.tensor(x, device=self.train_device, dtype=torch.float32)
        # self.history * 4 if not self.dow
        x = self.feature_extractor(
            x.view(-1, self.channels, self.history * (200 // self.down_factor), 200 // self.down_factor))
        # x = F.relu6(self.feature_extractor(x.view(-1, 3 * 2500)))
        # print(x.shape)
        # x = F.relu6(self.hidden_layer(x))
        x = F.relu6(self.hidden_layer(x.view(-1, 1 * 68 * 18)))
        probs = F.log_softmax(self.output(x))
        val = self.value(x)
        return probs, val


class Agent(object):
    def __init__(self, n_actions, replay_buffer_size=50000,
                 batch_size=32, hidden_size=18, gamma=0.98, model_name="dqn_model",
                 history=3, down_sample=False, gray_scale=False):
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.policy_net = Policy(hidden=hidden_size, train_device=self.train_device,
                                 history=history, down_sample=down_sample, gray_scale=gray_scale)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayMemory(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.model_name = model_name
        self.policy_net.to(self.train_device)
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.values = []
        
    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    def _do_network_update(self):
        action_probs = torch.stack(self.action_probs, dim=0) \
            .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        values = torch.stack(self.values, 0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards, self.values = [], [], [], []
        # TODO: Compute critic loss and advantages (T3)
        # if self.policy.every_10:
        #     end_states = np.array(end_states)
        #     next_vals = torch.cat((values[1:], self.last_val))
        #     next_vals[end_states == 1] = 0
        #     delta = rewards + self.gamma * next_vals - values
        #     policy_update = torch.mean(-delta.detach() * action_probs)
        #     val_fn_update = torch.mean(-delta.detach() * values)
        #     # val_fn_update = torch.mean(-delta * values)
        if 1 ==0:
            pass
        else:
            next_vals = torch.cat((values[1:].view(-1), torch.tensor([0.], device=self.train_device)))
            delta = rewards + self.gamma * next_vals - values
            policy_update = torch.mean(-delta.detach() * action_probs)
            val_fn_update = torch.mean(-delta.detach() * values)
        # TODO: Compute the optimization term (T1, T3)
        total_update = val_fn_update + policy_update
        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        total_update.backward()
        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def load_model(self, path=""):
        if path:
            print("Loading model from the given path " + path)
            self.policy_net.load_state_dict(torch.load(path))
        elif os.path.isfile(self.model_name + ".pth"):
            print("Loading the model " + self.model_name + " from the same folder.")
            self.policy_net.load_state_dict(torch.load(self.model_name + ".pth"))
        else:
            print("Error: neither path to the model was not given, nor does the model "
                  + self.model_name + " exist in the current folder path")
            exit(1)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    @staticmethod
    def get_name():
        return "Best Name"

    def save_model(self, path="", epoch=None):

        model_name = self.model_name + "_epoch_no_" + str(epoch) if epoch is not None else self.model_name
        if path:
            if path[-4:] == ".pth":
                print("Full path and name were given for the model. Saving it in " + path)
                torch.save(self.policy_net.state_dict(), path)
            else:
                print("Path without the file name were given. Saving it as " + path + self.model_name + ".pth")
                torch.save(self.policy_net.state_dict(), path + "/" + model_name + ".pth")
        else:
            print("No path was given. Saving it in the same folder as this script, with the name "
                  + model_name + ".pth")
            torch.save(self.policy_net.state_dict(), model_name + ".pth")

    def reset(self):
        print("What the hell is reset on the agent supposed to do..? xD Isn't the env supposed to be outside the agent"
              "class?")
        pass

    def get_action(self, observation, evaluation=False):

        # TODO: Pass state x through the policy network (T1)
        act_prob, val = self.policy_net(observation)
        act = torch.exp(act_prob).multinomial(num_samples=1).data[0]
        return act, act_prob[0, act[0]], val

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_outcome(self, observation, action_prob, reward, value, end_state=None):
        # print(len(self.states))
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.values.append(value)
        if end_state is not None:
            self.end_states.append(end_state)
