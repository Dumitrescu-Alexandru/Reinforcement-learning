import torch


def discount_rewards(r, gamma, done=None):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        if done is not None and done[t]:
            running_add = 0
        discounted_r[t] = running_add

    return discounted_r
