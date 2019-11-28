#%%
import sys
from pathlib import Path
parent = Path(__file__).parent
sys.path.append(str(parent.resolve()))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import NNPolicy, SharedAdam
from scipy.signal import lfilter

#%%
class Agent(object):
    def __init__(
        self,
        model_name="a3c_model",
        hidden_size=256,
        num_actions=3,
        save_dir=Path("."),
        base=True,
        lr=1e-4,
        test=True
    ):
        print("Creating Agent")
        self.save_dir = save_dir
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.lr = lr
        self.base = base
        self.test= test
        if self.base:
            # share memory if it is the base model
            self.policy = NNPolicy(
                channels=1, memsize=self.hidden_size, num_actions=self.num_actions
            ).share_memory()
            self.optimizer = SharedAdam(self.policy.parameters(), lr=self.lr)
        else:
            self.policy = NNPolicy(
                channels=1, memsize=self.hidden_size, num_actions=self.num_actions
            )
        self.hx = torch.zeros(1, hidden_size,requires_grad=True)

    def load_model(self,model_no=None):
        self.policy.try_load(self.save_dir,model_no)

    @staticmethod
    def get_name():
        return "AAAC"

    def save_model(self, num_frames=None):

        torch.save(
            self.policy.state_dict(),
            self.save_dir / f"model.{num_frames/ 1e6:.0f}.tar",
        )

    def reset(self):
        self.hx = torch.zeros(1, self.hidden_size,requires_grad=True)

    def get_action(self, state):
        if self.test:
            state = torch.tensor(self.preprocess(state))
            self.hx.detach()
        value, logit, self.hx = self.policy((state.view(1, 1, 50, 50), self.hx))
        logp = F.log_softmax(logit, dim=-1)
        if self.test:
            action = logp.max(1)[1].data
            # return only action if testing
            return action.numpy()[0]
        else:
            action = torch.exp(logp).multinomial(num_samples=1).data[0]
            # return value, log_prob, action if training
            return value, logp, action

    
    def preprocess(self,state_):
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
        small_image = (
            new_state.reshape((1, output_size, bin_size, output_size, bin_size))
            .max(4)
            .max(2)
        )
        return small_image

    @staticmethod
    def discount(x, gamma):
        return lfilter([1],[1,-gamma],x[::-1])[::-1]

    
    def cost_func(self,args, values, logps, actions, rewards):
        np_values = values.view(-1).data.numpy()

        # generalized advantage estimation using \delta_t residuals (a policy gradient method)
        delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
        logpys = logps.gather(1, torch.tensor(actions).view(-1,1))
        gen_adv_est = self.discount(delta_t, args.gamma * args.tau)
        policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()
        
        # l2 loss over value estimator
        rewards[-1] += args.gamma * np_values[-1]
        discounted_r = self.discount(np.asarray(rewards), args.gamma)
        discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
        value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

        entropy_loss = (-logps * torch.exp(logps)).sum() # entropy definition, for entropy regularization
        return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
    
# %%
