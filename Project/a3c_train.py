# based on Baby Advantage Actor-Critic by Sam Greydanus

from __future__ import print_function
import torch, os, gym, time, glob, argparse, sys
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import wimblepong
from test_bench.test_agents.a3c.agent import Agent
from pathlib import Path

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
    parser.add_argument("--save_dir", type=Path,default=Path("models/"),help="Save directory location")
    return parser.parse_args()




def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; f=(args.save_dir/'log.txt').open(mode) ; f.write(s+'\n') ; f.close()

def train(base_agent, rank, args, info):
    env = gym.make("WimblepongVisualSimpleAI-v0")
    env.unwrapped.scale = args.scale
    env.unwrapped.fps = args.fps

    local_agent = Agent(base=False,test=args.test,save_dir=args.save_dir) # a local/unshared model
    local_model = local_agent.policy
    state = torch.tensor(local_agent.preprocess(env.reset())) # get first frame

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping variables

    win1 =0
    while info['frames'][0] <= 8e7 or args.test: # train till 80M frames
        local_model.load_state_dict(base_agent.policy.state_dict()) # sync with shared model
        if done:
            # resets the hidden vector for RNN
            local_agent.reset()
        else :
            local_agent.hx = local_agent.hx.detach()

        values, logps, actions, rewards = [], [], [], [] # save values for computing gradientss

        for step in range(args.rnn_steps):
            episode_length += 1

            value, logp, action = local_agent.get_action(state)
            state, reward, done, _ = env.step(action.numpy()[0])

            if reward == 10 : win1+=1
            
            if args.render: env.render()
            
            reward = 0.01 if not done else reward
            state = torch.tensor(local_agent.preprocess(state)) ; epr += reward
            done = done or episode_length >= 1e4 # reduce total play time
            
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

            if done:
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

        loss = local_agent.cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        base_agent.optimizer.zero_grad() ; loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40)

        for param, shared_param in zip(local_model.parameters(), base_agent.policy.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad # sync gradients with shared model
        base_agent.optimizer.step()

if __name__ == "__main__":
    if sys.version_info[0] > 2:
        ctx = mp.get_context('spawn') # get spawn context for multi processing
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux!" # or else you get a deadlock in conv2d layer
    
    args = get_args()
    
    if args.render:  args.processes = 1 ; args.test = True # render mode -> test mode using one process
    if args.test:  args.lr = 0 # don't train in render mode
    args.num_actions = 3 # action space of pong

    if not args.save_dir.exists():
        args.save_dir.mkdir()

    torch.manual_seed(args.seed)
    base_agent = Agent(test=args.test,save_dir=args.save_dir)

    # create shared variable for training stats
    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    
    info['frames'] += base_agent.policy.try_load(args.save_dir) * 1e6
    
    if int(info['frames'].item()) == 0: printlog(args,'', end='', mode='w') # clear log file
    
    processes = []
    for rank in range(args.processes):
        p = ctx.Process(target=train, args=(base_agent, rank, args, info))
        p.start() ; processes.append(p)
    # join all threads
    for p in processes: p.join()