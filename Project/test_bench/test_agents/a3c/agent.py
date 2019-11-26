#%%
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
from model import NNPolicy
import glob
#%%
class Agent(object):
    def __init__(self, model_name="a3c_model",hidden_size = 256, num_actions=3,save_dir="./"):
        self.save_dir = save_dir
        self.num_frames = 0
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.policy = NNPolicy(channels=1, memsize=self.hidden_size, num_actions=self.num_actions).share_memory()
        self.hx = torch.zeros(1,hidden_size)
    
    def load_model(self):
        self.policy.try_load(self.save_dir)
    
    @staticmethod
    def get_name():
        return "AAAC"

    def save_model(self, epoch=None):

        torch.save(self.policy.state_dict(), self.save_dir+'model.{:.0f}.tar'.format(self.num_frames/1e6))

    def reset(self):
        self.hx = torch.zeros(1,self.hidden_size)
        self.num_frames = 0

    def get_action(self, state):
        state = torch.tensor(self.preprocess(state))
        self.hx.detach()
        value, logit, self.hx = self.policy((state.view(1,1,50,50), self.hx))
        logp = F.log_softmax(logit, dim=-1)
        action = logp.max(1)[1].data
        return action.numpy()[0]

    def preprocess(self, state_):
        # always grayscale
        if len(state_.shape) == 3:
            state_ = self.black_and_white(state_)

        return state_.astype(np.float32)

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
        return small_image



# %%
