# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

from __future__ import print_function
import torch, os, gym, time, glob, argparse, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import wimblepong
from test_bench.test_agents.a3c.agent import Agent

os.environ['OMP_NUM_THREADS'] = '1'

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--processes', default= 20, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    parser.add_argument("--fps", type=int, help="FPS for rendering", default=60)
    parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
    parser.add_argument("--save_dir", type=str,default="models/",help="Save directory location")
    return parser.parse_args()


discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner


def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; f=open(args.save_dir+'log.txt',mode) ; f.write(s+'\n') ; f.close()

def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, torch.tensor(actions).view(-1,1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()
    
    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

    entropy_loss = (-logps * torch.exp(logps)).sum() # entropy definition, for entropy regularization
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

def train(base_agent, rank, args, info):
    env = gym.make("WimblepongVisualSimpleAI-v0")
    env.unwrapped.scale = args.scale
    env.unwrapped.fps = args.fps

    local_agent = Agent(base=False,test=args.test,save_dir=args.save_dir) # a local/unshared model
    local_model = local_agent.policy
    state = torch.tensor(local_agent.preprocess(env.reset())) # get first state

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping

    win1 =0
    while info['frames'][0] <= 8e7 or args.test: # openai baselines uses 40M frames...we'll use 80M
        local_model.load_state_dict(base_agent.policy.state_dict()) # sync with shared model
        if done:
            # resets the hidden vector for RNN
            local_agent.reset()

        values, logps, actions, rewards = [], [], [], [] # save values for computing gradientss

        for step in range(args.rnn_steps):
            episode_length += 1

            value, logp, action = local_agent.get_action(state)
            state, reward, done, _ = env.step(action.numpy()[0])

            if reward == 10 : win1+=1
            
            if args.render: env.render()
            
            reward = 0.01 if not done else reward
            state = torch.tensor(local_agent.preprocess(state)) ; epr += reward
            done = done or episode_length >= 1e4 # don't playing one ep for too long
            
            info['frames'].add_(1) ; num_frames = int(info['frames'].item())
            
            if num_frames % 2e6 == 0: # save every 2M frames
                printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
                base_agent.save_model(num_frames)

            if done: # update shared data
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                info['run_epr'].mul_(1-interp).add_(interp * epr)
                info['run_loss'].mul_(1-interp).add_(interp * eploss)

            if rank == 0 and time.time() - last_disp_time > 60: # print info ~ every minute
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                printlog(args, 'time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
                    .format(elapsed, info['episodes'].item(), num_frames/1e6,
                    info['run_epr'].item(), info['run_loss'].item()))
                last_disp_time = time.time()

            if done: # maybe print info.
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(local_agent.preprocess(env.reset()))
                print("episode {} over. Broken WR: {:.3f} Loss: {:.3f}".format(
                    info['episodes'].item(),
                    win1/(info['episodes'].item()+1),
                    info['run_loss'].item()
                    ))
                if args.test and info['episodes'].item() % 5 == 4:
                    env.switch_sides()

            values.append(value) ; logps.append(logp) ; actions.append(action) ; rewards.append(reward)

        next_value = torch.zeros(1,1) if done else local_agent.get_action(state)[0]
        values.append(next_value.detach())

        loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        base_agent.optimizer.zero_grad() ; loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40)

        for param, shared_param in zip(local_model.parameters(), base_agent.policy.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad # sync gradients with shared model
        base_agent.optimizer.step()

if __name__ == "__main__":
    if sys.version_info[0] > 2:
        ctx = mp.get_context('spawn') # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux!" # or else you get a deadlock in conv2d
    
    args = get_args()
    
    # args.save_dir = '{}/'.format('models') # keep the directory structure simple
    if args.render:  args.processes = 1 ; args.test = True # render mode -> test mode w one process
    if args.test:  args.lr = 0 # don't train in render mode
    args.num_actions = 3 # get the action space of this game
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None # make dir to save models etc.

    torch.manual_seed(args.seed)
    base_agent = Agent(test=args.test,save_dir=args.save_dir)
    # shared_model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions).share_memory()
    # shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    
    info['frames'] += base_agent.policy.try_load(args.save_dir) * 1e6
    
    if int(info['frames'].item()) == 0: printlog(args,'', end='', mode='w') # clear log file
    
    processes = []
    for rank in range(args.processes):
        p = ctx.Process(target=train, args=(base_agent, rank, args, info))
        p.start() ; processes.append(p)
    for p in processes: p.join()