
from main_hypergrid import run_train
import os
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--experiment_name", type=str,
                    default="replay_and_capacity")
parser.add_argument("--folder", type=str, default="logs")
args = parser.parse_args()

experiment_dir = os.path.join(os.getcwd(), args.folder , args.experiment_name)
for name in os.listdir(experiment_dir):  # [::-1]:
    best_config_dir = os.path.join(experiment_dir, name)
    with open(os.path.join(best_config_dir + "/best_config.json"), 'r') as fp:
        config = json.load(fp)
        # Rerun for 3 different seeds
        for seed in range(100,103):
            config["seed"] = seed
            try:
                run_train(config,args.folder,use_wandb=True,im_show=False)
            except:
                print("Error in ", config)
                continue