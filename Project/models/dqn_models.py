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
    def __init__(self, train_device=torch.device("cuda:0"), channels=3):
        super(FeatExtractConv, self).__init__()
        self.train_device = train_device
        self.conv1 = nn.Conv2d(channels, 1, kernel_size=(7, 7), stride=1)
        self.max_pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=(5, 5), stride=2)
        self.max_pool2 = nn.MaxPool2d(2, stride=2)
        # self.conv3 = nn.Conv2d(3, 3, kernel_size=(5, 5), stride=2)
        # self.max_pool3 = nn.MaxPool2d(2, stride=2)
        self.train_device = train_device

    def forward(self, x):
        x = x.to(self.train_device)
        x = F.relu6(self.max_pool1(self.conv1(x)))
        x = F.relu6(self.max_pool2(self.conv2(x)))
        # x = F.relu6(self.max_pool3(self.conv3(x)))
        return x


class DQN(nn.Module):
    def __init__(self, hidden=18, fine_tune=True, train_device=torch.device("cuda:0"),
                                                    history=3, down_sample=False, gray_scale=False):
        super(DQN, self).__init__()
        self.train_device = train_device
        self.hidden = hidden
        self.history = history
        channels = 1 if gray_scale else 3
        down_factor = 4 if down_sample else 1
        self.feature_extractor = FeatExtractConv(channels, down_sample)
        if not fine_tune:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        self.hidden_layer = nn.Linear(1 * 17 * 4 * 4//down_sample, hidden)
        self.output = nn.Linear(hidden, 3)

    def forward(self, x):
        x = x.to(self.train_device)
        print(x.shape)
        #self.history * 4 if not self.dow
        x = self.feature_extractor(x.view(-1, 1, 150, 50))
        x = x.view(-1, 1 * 17 * 4)
        x = F.relu6(self.hidden_layer(x))
        x = self.output(x)
        return x


class Agent(object):
    def __init__(self, n_actions, replay_buffer_size=50000,
                 batch_size=32, hidden_size=18, gamma=0.98, model_name="dqn_model",
                 history=3, down_sample=False, gray_scale=False):
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available`() else "cpu")
        self.n_actions = n_actions
        self.policy_net = DQN(hidden_size, self.train_device,
                                        history, down_sample, gray_scale)
        self.target_net = DQN(hidden_size, self.train_device,
                                        history, down_sample, gray_scale)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-4)
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

    def get_action(self, state, epsilon=0.05):
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                state = torch.from_numpy(state).float()
                q_values = self.policy_net(state)
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
