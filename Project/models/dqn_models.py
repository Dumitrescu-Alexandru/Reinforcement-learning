import numpy as np
from collections import namedtuple
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from utils import Transition, ReplayMemory
import torchvision.models as models
from time import time


class FeatExtractConv(nn.Module):
    def __init__(self, train_device=torch.device("cuda:0"), channels=3, model_variant=1):
        super(FeatExtractConv, self).__init__()
        self.train_device = train_device
        self.model_variant = model_variant
        # # The previous architecture
        if self.model_variant == 1:
            self.conv1 = nn.Conv2d(channels, 32, kernel_size=(7, 7), stride=1)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=1)
            # self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1)
        elif self.model_variant == 2:
            # The one taken from a3c
            self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.train_device = train_device

    def forward(self, x):
        # For the previous architecture
        if self.model_variant == 1:
            x = F.relu6(self.conv1(x))
            x = F.max_pool2d(x, kernel_size=2)
            x = F.relu6(self.conv2(x))
            x = F.max_pool2d(x, kernel_size=2)
            # x = F.relu6(self.conv3(x))
            
        # for the borrowed thing from a3c
        elif self.model_variant == 2:
            x = x.to(self.train_device)
            x = F.relu6(self.conv1(x))
            x = F.relu6(self.conv2(x))
            x = F.relu6(self.conv3(x))
            x = F.relu6(self.conv4(x))

        return x


class DQN(nn.Module):
    def __init__(self, hidden=100, fine_tune=True, train_device=torch.device("cuda:0"),
                 history=3, down_sample=False, gray_scale=False, model_variant=1, train_=False,
                 channel_stack=False):
        super(DQN, self).__init__()
        self.train_device = train_device
        self.model_variant = model_variant
        self.hidden = hidden
        self.history = history
        self.channels = 1 if gray_scale else 3
        self.down_factor = 4 if down_sample else 1
        self.channels = self.channels if not channel_stack else self.channels * self.history
        self.feature_extractor = FeatExtractConv(channels=self.channels, model_variant=model_variant)
        # self.feature_extractor = nn.Linear(3 * 2500, hidden)
        if not fine_tune:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
        if self.model_variant == 1 or self.model_variant == 2:

            with torch.no_grad():
                if channel_stack:
                    out_dims = self.feature_extractor(torch.randn(1, self.history, 50, 50)).shape
                else:
                    out_dims = self.feature_extractor(torch.randn(1, 1, self.history * 50, 50)).shape
            out_dim = 1
            for o_d in out_dims:
                out_dim = out_dim * o_d
        elif self.model_variant == 2:
            out_dim = 32 * 10 * 4
        self.out_dim = out_dim
        self.dropout = nn.Dropout(p=0.2)
        self.hidden_layer = nn.Linear(out_dim, hidden)
        self.output = nn.Linear(hidden, 3)
        self.train_ = train_

    def forward(self, x):
        x = x.to(self.train_device)
        # self.history * 4 if not self.dow
        # x = self.feature_extractor(
        #     x.view(-1, self.channels, self.history * (200 // self.down_factor), 200 // self.down_factor))
        x = F.relu6(self.feature_extractor(x))
        x = F.relu6(self.hidden_layer(x.view(-1, self.out_dim)))

        if self.train_:
            x = self.dropout(x)
        # x = F.relu6(self.hidden_layer(x.view(-1, 2 * 48 * 15)))
        x = self.output(x)
        return x


class Agent(object):
    def __init__(self, n_actions, replay_buffer_size=50000,
                 batch_size=32, hidden_size=18, gamma=0.99, model_name="dqn_model",
                 history=3, down_sample=False, gray_scale=False, lr=1e-5, model_variant=1, train_=False,
                 channel_stack=False):
        self.lr = lr
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.train_ = train_
        self.policy_net = DQN(hidden=hidden_size, train_device=self.train_device,
                              history=history, down_sample=down_sample, gray_scale=gray_scale,
                              model_variant=model_variant, train_=train_, channel_stack=channel_stack)
        self.target_net = DQN(hidden=hidden_size, train_device=self.train_device,
                              history=history, down_sample=down_sample, gray_scale=gray_scale,
                              model_variant=model_variant, train_=train_, channel_stack=channel_stack)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr)
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.model_name = model_name
        self.policy_net.to(self.train_device)
        self.target_net.to(self.train_device)

    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = ~torch.tensor(batch.done)
        non_final_next_states = [s for nonfinal, s in zip(non_final_mask,
                                                          batch.next_state) if nonfinal > 0]
        non_final_next_states = torch.stack(non_final_next_states)
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action).to(self.train_device)
        reward_batch = torch.cat(batch.reward).to(self.train_device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size).to(self.train_device)
        next_state_values[non_final_mask.bool()] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Task 4: TODO: Compute the expected Q values
        expected_state_action_values = self.gamma * next_state_values + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(),
                                expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.requires_grad:
                param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def freeze_feat_extractor(self):
        self.policy_net.feature_extractor.requires_grad = False

    def load_conv(self, path):
        if path and ".pth" in path:
            print("Loading model from the given path " + path)
            model_dict = self.policy_net.state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "feature_extractor" in k}
            model_dict.update(pretrained_dict)
            self.policy_net.load_state_dict(model_dict)

    def load_model(self, path=""):
        if path and ".pth" in path:
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

    def get_action(self, state, epsilon=0.05):
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                state = torch.from_numpy(state).float()
                q_values = self.policy_net(state.unsqueeze(0))
                return torch.argmax(q_values).item()
        else:
            return random.randrange(self.n_actions)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.from_numpy(next_state).float()
        state = torch.from_numpy(state).float()
        self.memory.push(state, action, next_state, reward, done)
