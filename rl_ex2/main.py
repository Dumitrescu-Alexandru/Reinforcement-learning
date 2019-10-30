import numpy as np
from time import sleep
from sailing import SailingGridworld
import matplotlib.pyplot as plt
import seaborn as sns
# Set up the environment
env = SailingGridworld(rock_penalty=-2)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)

if __name__ == "__main__":
    # Reset the environment
    state = env.reset()

    # Compute state values and the policy
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))

    def sum_state_reward(transition, val_est):
        sum_on_action = 0
        for s_r in transition:
            next_state = s_r[0]
            rwd = s_r[1]
            dn = s_r[2]
            p = s_r[3]
            if dn:
                sum_on_action += p * rwd
            else:
                sum_on_action += p * (rwd + 0.9 * val_est[next_state[0], next_state[1]])
        return sum_on_action


    def calc_val_estimate(transitions, val_est):
        max_ = sum_state_reward(transitions[0], val_est)
        best_act = 0
        for k in range(1, len(transitions)):
            if sum_state_reward(transitions[k], val_est) > max_:
                best_act = k
                max_ = sum_state_reward(transitions[k], val_est)
        return max_, best_act


    def calc_val_estimates(use_delta=False, render=False):
        # TASK 1
        ##################################################################
        if not use_delta:
            for _ in range(50):
                if render:
                    env.clear_text()
                    env.render()
                    env.draw_values(value_est)
                for i in range(env.w):
                    for j in range(env.h):
                        value_est[i, j], best_a = calc_val_estimate(env.transitions[i, j], value_est)
                        policy[i, j] = best_a
        ##################################################################

        # TASK 3
        ##################################################################
        else:
            eps = 0.0001
            converged = False
            while not converged:
                if render:
                    env.render()
                    env.draw_values(value_est)
                    env.clear_text()
                converged = True
                for i in range(env.w):
                    for j in range(env.h):
                        val = value_est[i, j]
                        value_est[i, j], best_a = calc_val_estimate(env.transitions[i, j], value_est)
                        policy[i, j] = best_a
                        # implementation is equivalent to taking the max of the difference between subsequent
                        # value functions on an iteration on all the states (if at least one |delta| > eps, the loop
                        # doesn't stop
                        if np.abs(val - value_est[i, j]) > eps:
                            converged = False
        ##################################################################
    calc_val_estimates(use_delta=False, render=True)
    # Show the values and the policy
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()
    sleep(1)
    # Save the state values and the policy
    fnames = "values.npy", "policy.npy"
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)

    done = False
    # TASK 2
    ##################################################################
    while not done:
        # Select a random action
        action = int(policy[state])

        # Step the environment
        state, reward, done, _ = env.step(action)
        # Render and sleep
        env.draw_actions(policy)
        env.render()
        sleep(0.5)
    ##################################################################
    start_state = env.reset()

    G = []
    # TASK 4
    ##################################################################
    for i in range(1000):
        done = False
        time = 0
        policy_estimate = 0
        state = start_state
        while not done:
            action = int(policy[state])
            state, reward, done, _ = env.step(action)
            policy_estimate += 0.9 ** time * reward
            time += 1
        G.append(policy_estimate)
        start_state = env.reset()
    ##################################################################
    np.load()
    print("Mean of the first state's estimation taken from the policy: ", np.mean(G))
    print("Standard deviation of the first state's estimation:", np.std(G))
    print("Value function estimation", value_est[start_state[0], start_state[1]])
