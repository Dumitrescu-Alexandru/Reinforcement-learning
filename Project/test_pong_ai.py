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
parser.add_argument("--use_black_white", default=False, action="store_true")

args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

# Number of episodes/games to play
episodes = 100000
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
               batch_size=args.batch_size, hidden_size=12, gamma=0.98)
if args.load_model:
    player.load_model(args.load_model)
glie_a = 10000
avg_ttd = []


def black_and_white(state_):
    new_state = np.zeros((state_.shape[0], state_.shape[1]))

    def is_background(pxl):
        if pxl[0] == 43 and pxl[1] == 48 and pxl[2] == 58:
            return True
        return False

    for ind_1, col in enumerate(state_):
        for ind_2, pixel in enumerate(col):
            new_state[ind_1, ind_2] = 0 if is_background(pixel) else 1
    return new_state.reshape(1, new_state.shape[0], new_state.shape[1])


for i in range(0, episodes):
    done = False
    state = env.reset()
    eps = glie_a / (glie_a + i)
    cum_reward = 0
    state_list = []
    ttd = 0
    channels = 1 if args.use_black_white else 3
    while not done:
        ttd += 1
        # action1 is zero because in this example no agent is playing as player 0
        if args.use_black_white:
            if state.shape[-1] == 3 or state.shape[0] == 3:
                state = black_and_white(state)
                state_list.append(state)
        else:
            state_list.append(state.transpose(2, 0, 1))
        if len(state_list) >= 3:
            augmented_state = np.hstack(state_list[-3:]).reshape(channels, 600, 200)
            state_list = state_list[-3:]
        else:
            augmented_state = np.hstack((state, state, state)).reshape(channels, 600, 200)
        action = player.get_action(augmented_state, eps)  # player.get_action()
        next_state, rew1, done, info = env.step(action)
        next_state = black_and_white(next_state) if args.use_black_white else next_state
        rew1 = -50 / ttd if rew1 == 0 and done else rew1
        cum_reward += rew1
        augmented_state_next_state = augmented_state
        augmented_state_next_state[:, :400, :] = augmented_state[:, 200:, :]
        augmented_state_next_state[:, 400:, :] = next_state.reshape(channels, 200, 200)
        player.store_transition(augmented_state, action, augmented_state_next_state, rew1, done)
        player.update_network()
        if args.housekeeping:
            states.append(ob1)
        # Count the wins
        if rew1 == 10:
            win1 += 1
        # if not args.headless:
        #     env.render()
        if done:
            observation = env.reset()
            plt.close()  # Hides game window
            if args.housekeeping:
                plt.plot(states)
                plt.legend(["Player", "Opponent", "Ball X", "Ball Y", "Ball vx", "Ball vy"])
                plt.show()
                states.clear()
            print("episode {} over. Broken WR: {:.3f}".format(i, win1 / (i + 1)))
            avg_ttd.append(ttd)
            if len(avg_ttd) > 50:
                del (avg_ttd[:len(avg_ttd) - 50])
            if len(avg_ttd) == 50:
                print("episode {} over. Time till death (Average on last 50): {}".format(i, sum(avg_ttd) / 50))
            if i % 5 == 4 and args.switch_sides:
                env.switch_sides()
        state = next_state
        if i % TARGET_UPDATE == 0:
            player.update_target_network()
    if i % args.save_every == 0 and i != 0:
        print("HM")
        player.save_model(epoch=i)
