"""
This is an example on how to use the two player Wimblepong environment with one
agent and the SimpleAI
"""
import matplotlib.pyplot as plt
from time import sleep
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
import torch
from models.dqn_models import Agent
import pickle
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--housekeeping", action="store_true",
                    help="Plot, player and ball positions and velocities at the end of each episode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=60)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument("--save_every", type=int, default=1000, help="Save every n number of episodes")
parser.add_argument("--load_model", type=str, default="", help="Load some model")
parser.add_argument("--replay_buffer_size", type=int, default=5000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--switch_sides", default=False, action="store_true")
parser.add_argument("--use_black_white", default=True, action="store_true")
parser.add_argument("--down_sample", default=True, action="store_true")
parser.add_argument("--save_dir", default='./', type=str, help="Directory to save models")
parser.add_argument("--history", default=3, type=int, help="Number of previous frames in state")
parser.add_argument("--step_multiple", default=1, type=int, help="Number times to step with the same action")
parser.add_argument("--test", default=False, action="store_true", help="Freeze the model and set eps to 0")
parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate for the optimizer")
parser.add_argument("--reward_type", default="incentivise_ttd", type=str, help="Learning rate for the optimizer")
parser.add_argument("--glie_a", default=2000000, type=int, help="Learning rate for the optimizer")
parser.add_argument("--model_variant", default=1, type=int, help="Index of the model")
parser.add_argument("--hidden_size", default=18, type=int, help="Hidden size of the DQN model")
parser.add_argument("--load_conv", default=False, action="store_true",
                    help="Loads conv extractor from some other model")
parser.add_argument("--fixed_eps", default=-1, type=float, help="set a fixed epsilon-exploration parameter")
parser.add_argument("--dropout", default=False, action="store_true", help="set dropout to the DQN model")
parser.add_argument("--channels_stack", default=False, action="store_true", help="Set history images to channels")
parser.add_argument("--start_episode", default=0, type=int, help="Start from specified episode")
parser.add_argument("--freeze_feat_extractor", default=False, action="store_true",
                    help="Freeze the feature extractor part of the model for parameter tuning")
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

# Number of episodes/games to play
episodes = 170000
TARGET_UPDATE = 20
# Define the player
player_id = 1
# Set up the player here. We used the SimpleAI that does not take actions for now
# player = wimblepong.SimpleAi(env, player_id)

# Housekeeping
states = []
win1 = 0
ob1 = env.reset()
player = Agent(n_actions=3, replay_buffer_size=args.replay_buffer_size,
               batch_size=args.batch_size, hidden_size=args.hidden_size, gamma=0.98, history=args.history,
               down_sample=args.down_sample, gray_scale=args.use_black_white, lr=args.lr,
               model_variant=args.model_variant, train_=args.dropout and not args.test,
               channel_stack=args.channels_stack)
if args.load_conv:
    player.load_conv(args.load_model)
    player.freeze_feat_extractor()
elif args.load_model:
    player.load_model(args.load_model)
if args.freeze_feat_extractor:
    player.freeze_feat_extractor()
glie_a = args.glie_a
avg_ttd = []
avg_wr = []


def get_reward(rew, dn, t):
    if args.reward_type == "incentivise_ttd":
        return 0.01 if rew == 0 and not dn else rew
    elif args.reward_type == "anneal_incentivise_ttd":
        return 0.995 ** ((t - 50) ** 2) if rew == 0 and not dn else rew
    elif args.reward_type == "unchanged":
        return rew


def black_and_white(state_, black_right_pallet=False):
    # grayscale weights for rgb
    gray_weights = [0.299, 0.587, 0.114]
    new_state = np.dot(state_, gray_weights)
    new_state = new_state.reshape(1, new_state.shape[0], new_state.shape[1])
    if args.down_sample:
        input_size = 200
        output_size = 50
        bin_size = input_size // output_size
        small_image = new_state.reshape((1, output_size, bin_size,
                                         output_size, bin_size)).max(4).max(2)
        small_image = small_image[0]
        if black_right_pallet:
            small_image[:, 45:50] = 47.645
        return small_image
        # return small_image[0]
    # send only a (size,size) image
    return new_state[0]


def preprocess(state_, black_right_pallet=True):
    if args.use_black_white:
        # if not already greyscale
        if len(state_.shape) == 3:
            state_ = black_and_white(state_, black_right_pallet)

    # make rgb channels as last dim instead of first
    elif state_.shape[0] == 3:
        print("Something is wrong with color images")
        state_ = state_.transpose(1, 2, 0)

    return state_


def augment(state_list, m=3):
    # augment the sates with m previous image states
    channels = 1 if args.use_black_white else 3

    img_size = state_list[-1].shape[1]

    # if there are m previous images
    if len(state_list) >= m:
        if args.channels_stack:
            augmented_state = np.array(state_list[-m:])
        else:
            augmented_state = np.vstack(state_list[-m:])
            # remove the old images
        state_list = state_list[-m:]

    # copy the last state m times
    else:
        first_state = state_list[0]
        temp = [first_state] * (m - len(state_list)) + state_list
        if args.channels_stack:
            augmented_state = np.array(temp)
        else:
            augmented_state = np.vstack(temp)

    # just add a channel in the beginning for torch to play nice
    if channels == 1 and not args.channels_stack:
        augmented_state = augmented_state[np.newaxis, :]
    return augmented_state


frames = 0
for i in range(args.start_episode, episodes):
    done = False
    state = env.reset()
    if args.test:
        eps = 0
    elif args.fixed_eps != -1:
        eps = args.fixed_eps
    elif args.start_episode != 0:
        eps = max(0.1, 1 - i / 180000)
    else:
        eps = max(0.3, (glie_a - frames) / glie_a)
    cum_reward = 0
    state_list = []
    # time till death for a single game 
    ttd = 0
    channels = 1 if args.use_black_white else 3
    black_right_pallet = True
    state_list.append(preprocess(state, black_right_pallet))
    if i % 15000 == 0 and i != 0:
        for param_group in player.optimizer.param_groups:
            param_group['lr'] = 1e-6
    while not done:
        ttd += 1
        # add the preprocessed image to list
        # get the history augmented state vector

        augmented_state = augment(state_list, args.history)
        if i == 3:
            plt.imshow(augmented_state.reshape(150, 50), cmap="gray")
            plt.show()
        action = player.get_action(augmented_state, eps)
        for _ in range(args.step_multiple):
            next_state, rew1, done, info = env.step(action)
            next_state = preprocess(next_state, black_right_pallet)
            state_list.append(next_state)
            cum_reward += rew1
            augmented_state_next_state = augment(state_list, args.history)
            rew1 = get_reward(rew1, done, ttd)
            player.store_transition(augmented_state, action, augmented_state_next_state, rew1, done)

            augmented_state = augmented_state_next_state

            if not args.headless:
                env.render()
            if done:
                break

        # get the augmented next state from the list
        # store the values 
        if not args.test:
            player.update_network()
        if args.housekeeping:
            states.append(ob1)
        # Count the wins
        if rew1 == 10:
            win1 += 1
        if done:
            avg_wr.append(rew1 / 10)
            observation = env.reset()
            plt.close()  # Hides game window
            if args.housekeeping:
                plt.plot(states)
                plt.legend(["Player", "Opponent", "Ball X", "Ball Y", "Ball vx", "Ball vy"])
                plt.show()
                states.clear()
            print("episode {} over. Broken WR: {:.3f}. Epsilon {:.3f}".format(i, win1 / (i + 1), eps))

            avg_ttd.append(ttd * args.step_multiple)
            frames += (ttd * args.step_multiple)
            # only keeping the last 50 time to deaths
            if len(avg_wr) > 1000:
                del (avg_wr[:-1000])
            if len(avg_ttd) > 50:
                del (avg_ttd[:len(avg_ttd) - 50])
            if len(avg_ttd) == 50 or len(avg_wr) == 1000:
                print("episode {} over. Time till death (Average on last 50): {} and WR: {:.3f}".format(i, sum(
                    avg_ttd) / 50, np.mean(avg_wr)))

            if i % 5 == 4 and args.switch_sides:
                env.switch_sides()
        state = next_state
        if i % TARGET_UPDATE == 0 and not args.test:
            player.update_target_network()
    # save the model
    if i % args.save_every == 0 and i != 0:
        print("HM")
        player.save_model(epoch=i, path=args.save_dir)
