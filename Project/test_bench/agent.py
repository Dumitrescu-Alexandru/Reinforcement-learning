import numpy as np
from collections import namedtuple
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import torchvision.models as models
from time import time

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class FeatExtractConv(nn.Module):
    def __init__(self, train_device=torch.device("cuda:0"), channels=1):
        super(FeatExtractConv, self).__init__()
        self.train_device = train_device
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=(7, 7), stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=1)
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
        x = F.max_pool2d(x, kernel_size=2)
        # x = F.relu6(self.max_pool1(self.conv1(x)))
        # x = F.relu6(self.max_pool2(self.conv2(x)))
        # x = F.relu6(self.max_pool3(self.conv3(x)))
        return x


class DQN(nn.Module):
    def __init__(self, hidden=100, fine_tune=True, train_device=torch.device("cuda:0"),
                 history=3, down_sample=False, gray_scale=False):
        super(DQN, self).__init__()
        self.train_device = train_device
        self.hidden = hidden
        self.history = history
        # always 1 chanel (always grayscale)
        self.channels = 1
        self.down_factor = 4 if down_sample else 1
        self.feature_extractor = FeatExtractConv(channels=self.channels)
        # self.feature_extractor = nn.Linear(3 * 2500, hidden)
        if not fine_tune:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        self.hidden_layer = nn.Linear(32 * 34 * 9, hidden)
        self.output = nn.Linear(hidden, 3)

    def forward(self, x):
        x = x.to(self.train_device)
        # self.history * 4 if not self.dow
        # x = self.feature_extractor(
        #     x.view(-1, self.channels, self.history * (200 // self.down_factor), 200 // self.down_factor))
        x = F.relu6(self.feature_extractor(x))
        x = F.relu6(self.hidden_layer(x.view(-1, 32 * 34 * 9)))
        # x = F.relu6(self.hidden_layer(x.view(-1, 2 * 48 * 15)))
        x = self.output(x)
        return x


class Agent(object):
    def __init__(self, replay_buffer_size=50000,
                 batch_size=32, hidden_size=12, gamma=0.98, model_name="dqn_model",
                 history=3, down_sample=False, gray_scale=False):
        self.history = history
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(hidden=hidden_size, train_device=self.train_device,
                              history=history, down_sample=down_sample, gray_scale=gray_scale)
        self.state_list = []
        self.target_net = DQN(hidden=hidden_size, train_device=self.train_device,
                              history=history, down_sample=down_sample, gray_scale=gray_scale)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-5)
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
        self.state_list = []

    def get_action(self, state):
        self.state_list.append(self.preprocess(state))
        with torch.no_grad():
            augmented_state = self.augment(self.history)
            augmented_state = torch.from_numpy(augmented_state).float()
            q_values = self.policy_net(augmented_state.unsqueeze(0))
            return torch.argmax(q_values).item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.from_numpy(next_state).float()
        state = torch.from_numpy(state).float()
        self.memory.push(state, action, next_state, reward, done)

    def preprocess(self, state_):
        # always grayscale
        if len(state_.shape) == 3:
            state_ = self.black_and_white(state_)

        # make rgb channels as last dim instead of first
        elif state_.shape[0] == 3:
            print("Something is wrong with color images")
            state_ = state_.transpose(1, 2, 0)

        return state_

    @staticmethod
    def black_and_white(state_):
        # grayscale weights for rgb
        gray_weights = [0.299, 0.587, 0.114]
        new_state = np.dot(state_, gray_weights)
        new_state = new_state.reshape(1, new_state.shape[0], new_state.shape[1])
        # always downsample
        input_size = 200
        output_size = 50
        bin_size = input_size // output_size
        small_image = new_state.reshape((1, output_size, bin_size,
                                         output_size, bin_size)).max(4).max(2)
        return small_image[0]

    def augment(self, m=3):
        # augment the sates with m previous image states
        channels = 1

        img_size = self.state_list[-1].shape[1]

        # if there are m previous images
        if len(self.state_list) >= m:
            augmented_state = np.vstack(self.state_list[-m:])
            # remove the old images
            self.state_list = self.state_list[-m:]

        # copy the last state m times
        else:
            first_state = self.state_list[0]
            temp = [first_state] * (m - len(self.state_list)) + self.state_list
            augmented_state = np.vstack(temp)

        # just add a channel in the beginning for torch to play nice
        if channels == 3:
            augmented_state = augmented_state.tranpose(2, 0, 1)
        elif channels == 1:
            augmented_state = augmented_state[np.newaxis, :]
        return augmented_state
