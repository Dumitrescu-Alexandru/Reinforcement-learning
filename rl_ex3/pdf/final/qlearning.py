import gym
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 2

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# For LunarLander, use the following values:
#         [  x     y    xdot ydot theta  thetadot cl  cr
# s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
# s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]


# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = 20000 / 9  # TODO: Set the correct value.
initial_q = 0  # T3: Set to 50

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

# cartpole q_grid
# q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q
# lunar lander q_grid
q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q

# Training loop
ep_lengths, epl_avg = [], []


def get_discrete_state(state_):
    x_point = np.argmin(np.abs(x_grid - state_[0]))
    v_point = np.argmin(np.abs(v_grid - state_[1]))
    th_point = np.argmin(np.abs(th_grid - state_[2]))
    av_point = np.argmin(np.abs(av_grid - state_[3]))
    return x_point, v_point, th_point, av_point

for ep in range(episodes + test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    epsilon = a / (a + ep)  # T1: GLIE/constant, T3: Set to 0
    epsilon = 0
    while not done:
        take_greedy = np.random.rand() > epsilon
        # luner lander
        discrete_state = get_discrete_state(state)
        if take_greedy:
            action = np.argmax(q_grid[discrete_state])
        else:
            action = int(np.random.rand() * 2)
        new_state, reward, done, _ = env.step(action)
        if not test:
            discrete_new_state = get_discrete_state(new_state)
            next_state_val = np.max(q_grid[discrete_new_state])
            if done:
                next_state_val = 0
            q_grid[(*discrete_state, action)] = (1 - alpha) * q_grid[(*discrete_state, action)] + \
                                                alpha * (reward + gamma * next_state_val)

        else:
            env.render()
        state = new_state
        steps += 1

    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep - 500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep - 200):])))

# Save the Q-value array
np.save("q_values.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

# Calculate the value function
values = np.max(q_grid, 4)
np.save("value_func.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY

# Plot the heatmap
plt.imshow(np.mean(values, axis=(1, 3)))
plt.show()
# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()
