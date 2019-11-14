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

# Define the player
player_id = 1
# Set up the player here. We used the SimpleAI that does not take actions for now
# player = wimblepong.SimpleAi(env, player_id)

# Housekeeping
states = []
win1 = 0
ob1 = env.reset()
player = Agent(n_actions=3, replay_buffer_size=50000,
               batch_size=32, hidden_size=12, gamma=0.98)
glie_a = 50
for i in range(0, episodes):
    done = False
    state = env.reset()
    eps = glie_a / (glie_a + i)
    cum_reward = 0
    while not done:
        # action1 is zero because in this example no agent is playing as player 0
        action = player.get_action(state.reshape(-1, 3, 200, 200))  # player.get_action()
        next_state, rew1, done, info = env.step(action)
        cum_reward += rew1
        player.store_transition(state.reshape(3, 200, 200), action, next_state.reshape(3, 200, 200), rew1, done)
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
            if i % 5 == 4:
                env.switch_sides()
        state = next_state
