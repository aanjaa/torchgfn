import json
from argparse import ArgumentParser

import torch
import wandb
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.utils import trajectories_to_training_samples, validate
from scripts.configs import load_config, make_env, make_loss, make_optim, make_sampler
from tqdm import tqdm, trange
from scipy.stats import multivariate_normal

from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import os


import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
from train import train
from raytune import EXPERIMENT_NAMES
def plot(reward,states,im_show):
    reward = reward.numpy()
    x = states.states_tensor[:, :, 0]
    y = states.states_tensor[:, :, 1]

    # 2D plot
    fig2d, ax = plt.subplots()
    im = ax.imshow(reward, cmap=cm.gist_earth, origin="lower")
    fig2d.colorbar(im, ax=ax)
    fig2d.x_ticks = x
    fig2d.y_ticks = y
    if im_show:
        plt.show()

    # 3D plot
    fig3d, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ls = LightSource(270, 45)
    rgb = ls.shade(reward, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, reward, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)
    if im_show:
        plt.show()
    return fig2d,fig3d

def run_train(config,use_wandb,im_show):
    if use_wandb:
        wandb.init(project=config["experiment_name"], name=config["name"])
        wandb.config.update(config)

    train(config,args.use_wandb)

    env = make_env(config)
    states = env.build_grid()
    reward = env.reward(states)
    fig2d, fig3d = plot(reward, states, im_show)
    if use_wandb:
        wandb.log({"2d": wandb.Image(fig2d), "3d": wandb.Image(fig3d)})
        wandb.finish()
    del fig3d,fig2d


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="reward_losses") #smoothness_losses "reward_losses" "searchspaces_losses"
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--local_debug", type=bool, default=True)
    parser.add_argument("--use_wandb", type=bool, default=False)
    args = parser.parse_args()

    # Makes it easier to debug
    if args.local_debug:
        config = {'env': {'device': 'cpu',
                          'ndim': 2,
                          'height': 32,
                          'R0': 0.1,
                          'R1': 0.5,
                          'R2': 2.0,
                          'reward_name': 'gmm-grid',  # ["cos","gmm-grid","gmm-random","center","corner","default"]
                          'num_means': 4,
                          'cov_scale': 7.0,
                          'quantize_bins': 5,
                          'preprocessor_name': 'KHot',
                          'name': 'hypergrid'},
                  'loss': {'module_name': 'NeuralNet',
                           'n_hidden_layers': 2,
                           'hidden_dim': 256,
                           'activation_fn': 'relu',
                           'forward_looking': False,
                           'name': 'detailed-balance'
                           },
                  'optim': {'lr': 0.001, 'lr_Z': 0.1, 'betas': [0.9, 0.999], 'name': 'adam'},
                  'sampler': {'temperature': 1.0, 'sf_bias': 0.0, 'epsilon': 0.0},
                  'seed': 0,
                  'batch_size': 16,
                  'n_iterations': 101,  # 1001,
                  'replay_buffer_size': 0,
                  'no_cuda': False,
                  'name': 'debug',
                  'experiment_name': 'debug',
                  'validation_interval': 100,
                  'validation_samples': 200000,
                  'resample_for_validation': False}
        run_train(config, args.use_wandb, im_show=True)

    # Log to wandb tuned hyperparams
    else:
        experiment_dir = os.path.join(os.getcwd(), "logs", args.experiment_name)
        for name in os.listdir(experiment_dir):#[::-1]:
            best_config_dir = os.path.join(experiment_dir, name)
            with open(os.path.join(best_config_dir + "/best_config.json"), 'r') as fp:
                config = json.load(fp)
                # Rerun for 3 different seeds
                for seed in range(100,103):
                    config["seed"] = seed
                    run_train(config,args.use_wandb,im_show=False)


