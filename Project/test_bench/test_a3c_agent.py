import argparse
import sys
import os
from pong_testbench import PongTestbench
from matplotlib import font_manager
import importlib
from pathlib import Path
from tqdm import tqdm
import json


parser = argparse.ArgumentParser()
parser.add_argument("--dir1", type=str, help="Directory to agent 1 to be tested.")
parser.add_argument("--dir2", type=str, default=None, nargs="?",
                    help="Directory to agent 2 to be tested. If empty, SimpleAI is used instead.")
parser.add_argument("--render", "-r", action="store_true", help="Render the competition.")
parser.add_argument("--games", "-g", type=int, default=100, help="number of games.")


args = parser.parse_args()
test_agent_dir = Path(__file__).absolute().parent / "test_agents"

args.dir1 = str(test_agent_dir/"a3c")
sys.path.insert(0, args.dir1)
import agent

orig_wd = os.getcwd()
os.chdir(args.dir1)
model_numbers = range(2,52,2)

models_dir =  Path(__file__).absolute().parent.parent

model_types ={
    'normal_rewards':models_dir/"models_a3c_normal",
    '0.01_rewards' : models_dir/"models_a3c_2",
    '0.05_rewards' : models_dir/"models_a3c"
}

test_agent_dir = Path(__file__).absolute().parent / "test_agents"

opp_agents = {
    'SimpleAI':None,
    'KarpathyNotTrained':test_agent_dir/"KarpathyNotTrained",
    'SomeAgent':test_agent_dir/"SomeAgent",
    'SomeOtherAgent':test_agent_dir/"SomeOtherAgent"
}

max_wins = 0
data = {}
for variant in tqdm(model_types):
    data[variant] = {} 
    for model_no in tqdm(model_numbers):
        data[variant][model_no] = {}
        for opponent in tqdm(opp_agents):
            save_dir = model_types[variant]
            agent1 = agent.Agent()
            agent1.save_dir = save_dir
            agent1.load_model(model_no)

            os.chdir(orig_wd)
            # del sys.path[0]
            args.dir2 = opp_agents[opponent]

            if args.dir2:
                sys.path.insert(0, str(args.dir2))
                importlib.reload(agent)
                os.chdir(args.dir2)
                agent2 = agent.Agent()
                agent2.load_model()
                os.chdir(orig_wd)
                del sys.path[0]
            else:
                agent2 = None

            testbench = PongTestbench(args.render)
            testbench.init_players(agent1, agent2)
            w1, w2 = testbench.run_test(args.games)
            data[variant][model_no][opponent] = w1

with open('data.json', 'w') as fp:
    json.dump(data, fp)