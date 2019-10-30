import gym
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(123)

env = gym.make('LunarLander-v2')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 4

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
initial_q = 50  # T3: Set to 50

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

# cartpole q_grid
# q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q
# lunar lander q_grid
q_grid = np.zeros((discr, discr, discr, discr, discr, discr, 2, 2, num_of_actions)) + initial_q

# Training loop
ep_lengths, epl_avg = [], []


def get_discrete_state_lunar(state_):
    x_point = np.argmin(np.abs(x_grid - state_[0]))
    y_point = np.argmin(np.abs(v_grid - state_[1]))
    xdot_point = np.argmin(np.abs(th_grid - state_[2]))
    ydot_point = np.argmin(np.abs(av_grid - state_[3]))
    theta_point = np.argmin(np.abs(av_grid - state_[4]))
    theta_dot_point = np.argmin(np.abs(av_grid - state_[5]))
    return x_point, y_point, xdot_point, ydot_point, theta_point, theta_dot_point, int(state_[6]), int(state_[7])


rwd_list = []
rwd_500_avg_list = []
for ep in range(episodes + test_episodes):
    rwd_sum = 0
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    epsilon = a / (a + ep)  # T1: GLIE/constant, T3: Set to 0
    while not done:
        # TODO: IMPLEMENT HERE EPSILON-GREEDY
        take_greedy = np.random.rand() > epsilon
        discrete_state = get_discrete_state_lunar(state)
        if take_greedy:
            action = np.argmax(q_grid[discrete_state])
        else:
            action = int(np.random.rand() * 4)
        new_state, reward, done, _ = env.step(action)
        rwd_sum += reward
        if not test:
            # TODO: ADD HERE YOUR Q_VALUE FUNCTION UPDATE
            discrete_new_state = get_discrete_state_lunar(new_state)
            next_state_val = np.max(q_grid[discrete_new_state])
            q_grid[(*discrete_state, action)] = (1 - alpha) * q_grid[(*discrete_state, action)] + \
                                                alpha * (reward + gamma * next_state_val)
        else:
            env.render()
        state = new_state
        steps += 1
    rwd_list.append(rwd_sum)
    rwd_500_avg_list.append(np.mean(rwd_list[max(0, ep-500):]))
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep - 500):]))

    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep - 200):])))

plt.plot(rwd_list)
plt.plot(rwd_500_avg_list)
plt.legend(["Episode reward", "500 episode average"])
plt.title("Episode reward sum")
plt.show()
