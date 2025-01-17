# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class NNPolicy(nn.Module): # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 4 * 4, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        hx = self.gru(x.view(-1, 32 * 4 * 4), (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx

    def try_load(self, save_dir,model_no =None):
        if model_no is None:
            step =0
            if (save_dir/'model.mdl').exists():
                self.load_state_dict(torch.load(save_dir/'model.mdl'))
                step =1
            return step            
            # paths = list(save_dir.glob('*.mdl'))
            # step = 0
            # if len(paths) > 0:
            #     ckpts = [int(s.stem.split('.')[-1]) for s in paths]
            #     ix = np.argmax(ckpts) ; step = ckpts[ix]
            #     self.load_state_dict(torch.load(paths[ix]))
            # print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
            return step
        else:
            model_path = save_dir/f"model.{model_no}.mdl"
            if model_path.exists():
                self.load_state_dict(torch.load(model_path))
                step = model_no
            else:
                step = 0
            print(f"Problem model location\n {model_path} doesn't exist") if step is 0 else print("\tloaded model: {}            ".format(model_path.stem))
            return step

class SharedAdam(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)
