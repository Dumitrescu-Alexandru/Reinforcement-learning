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
args = parser.parse_args()
import matplotlib.pyplot as plt

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
player = Agent(n_actions=3, replay_buffer_size=50000,
               batch_size=16, hidden_size=12, gamma=0.98)
glie_a = 50
for i in range(0, episodes):
    done = False
    state = env.reset()
    eps = glie_a / (glie_a + i)
    cum_reward = 0
    state_list = []
    while not done:
        # action1 is zero because in this example no agent is playing as player 0
        state_list.append(state.reshape(3, 200, 200))
        if len(state_list) > 3:
            # print("np.hstack(state_list[-3:])", np.stack(state_list[-3:]).shape)
            # plt.imshow(state_list[-1].reshape(200,200,3))
            # plt.show()
            # plt.imshow(np.stack(state_list[-3:]).reshape(600,200 ,3))
            # plt.show()
            # print("np.concatenate(state_list[-3:])", np.concatenate(state_list[-3:]).shape)
            augmented_state = np.hstack(state_list[-3:]).reshape(3, 600, 200)
        else:
            augmented_state = np.stack((state, state, state)).reshape(3, 600, 200)
        action = player.get_action(augmented_state, eps)  # player.get_action()
        next_state, rew1, done, info = env.step(action)
        cum_reward += rew1
        augmented_state_next_state = augmented_state
        augmented_state_next_state[:, :400, :] = augmented_state[:, 200:, :]
        augmented_state_next_state[:, 400:, :] = next_state.reshape(3, 200, 200)
        # plt.imshow(augmented_state_next_state.reshape(600,200,3))
        # plt.show()
        # player.store_transition(state.reshape(3, 200, 200), action, next_state.reshape(3, 200, 200), rew1, done)
        player.store_transition(augmented_state, action, augmented_state_next_state, rew1, done)
        player.update_network()
        if args.housekeeping:
            states.append(ob1)
        # Count the wins
        if rew1 == 10:
            win1 += 1
        if not args.headless:
            env.render()
        if done:
            observation = env.reset()
            plt.close()  # Hides game window
            if args.housekeeping:
                plt.plot(states)
                plt.legend(["Player", "Opponent", "Ball X", "Ball Y", "Ball vx", "Ball vy"])
                plt.show()
                states.clear()
            print("episode {} over. Broken WR: {:.3f}".format(i, win1 / (i + 1)))
            if i % 5 == 4:
                env.switch_sides()
        state = next_state
        if i % TARGET_UPDATE == 0:
            player.update_target_network()
