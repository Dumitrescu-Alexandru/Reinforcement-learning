import matplotlib.pyplot as plt
import numpy as np
import pickle

f = open("haifu_results_someagent.bin", "rb")
# f150_160 = open("haifu_results150-160.bin", "rb")
ttds = []
wins = []
while (1):
    try:
        w1, w2, ttd = pickle.load(f)
        wins.append(w1)
        ttds.append(ttd)
    except:
        wins = np.array(wins)
        ttds = np.array(ttds)
        break
# while (1):
#     try:
#         w1, w2, ttd = pickle.load(f150_160)
#         wins.append(w1)
#         ttds.append(ttd)
#     except:
#         wins = np.array(wins)
#         ttds = np.array(ttds)
#         break

episode_no = np.array(list(range(1, 161)))
# plt.plot(episode_no, ttds/100)
# plt.plot(episode_no, wins)
plt.plot(episode_no, ttds/100)
plt.legend(["Number of time steps per episode"])
# plt.legend(["Number of wins against someAgent"])  # ",""])
plt.xlabel("Episode Number ( x 1000)")
plt.ylabel("Model performance")
plt.show()
