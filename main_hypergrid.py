from argparse import Namespace

import matplotlib.pyplot as plt
import torch
import wandb
from matplotlib import cm
from matplotlib.colors import LightSource
from tqdm import tqdm, trange

from gfn.containers.replay_buffer import ReplayBuffer
from gfn.estimators import LogEdgeFlowEstimator, LogStateFlowEstimator, LogZEstimator
from gfn.gym import HyperGrid
from gfn.losses import (
    DBParametrization,
    FMParametrization,
    LogPartitionVarianceParametrization,
    SubTBParametrization,
    TBParametrization,
)
from gfn.utils.common import trajectories_to_training_samples, validate
from gfn.utils.estimators import DiscretePBEstimator, DiscretePFEstimator
from gfn.utils.modules import DiscreteUniform, NeuralNet, Tabular
import numpy as np


def train_hypergrid(config, use_wandb):
    args = Namespace(**config)
    seed = args.seed if args.seed != 0 else torch.randint(int(2**32), (1,))[0].item()
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # 1. Create the environment
    env = HyperGrid(
        args.ndim, args.height, args.reward_type, args.R0, args.R1, args.R2, args.n_means, args.cov_scale,
        args.quantize_bins, device_str=device_str
    )

    # 2. Create the parameterization.
    #    For this we need modules and estimators.
    #    Depending on the loss, we may need several estimators:
    #       one (forward only) for FM loss,
    #       two (forward and backward) or other losses
    #       three (same, + logZ) estimators for TB.
    parametrization = None
    if args.loss == "FM":
        # We need a LogEdgeFlowEstimator
        if args.tabular:
            module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
        else:
            module = NeuralNet(
                input_dim=env.preprocessor.output_dim,
                output_dim=env.n_actions,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
            )
        estimator = LogEdgeFlowEstimator(env=env, module=module)
        parametrization = FMParametrization(estimator)
    else:
        pb_module = None
        # We need a DiscretePFEstimator and a DiscretePBEstimator
        if args.tabular:
            pf_module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
            if not args.uniform_pb:
                pb_module = Tabular(n_states=env.n_states, output_dim=env.n_actions - 1)
        else:
            pf_module = NeuralNet(
                input_dim=env.preprocessor.output_dim,
                output_dim=env.n_actions,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
            )
            if not args.uniform_pb:
                pb_module = NeuralNet(
                    input_dim=env.preprocessor.output_dim,
                    output_dim=env.n_actions - 1,
                    hidden_dim=args.hidden_dim,
                    n_hidden_layers=args.n_hidden,
                    torso=pf_module.torso if args.tied else None,
                )
        if args.uniform_pb:
            pb_module = DiscreteUniform(env.n_actions - 1)

        assert (
                pf_module is not None
        ), f"pf_module is None. Command-line arguments: {args}"
        assert (
                pb_module is not None
        ), f"pb_module is None. Command-line arguments: {args}"

        pf_estimator = DiscretePFEstimator(env=env, module=pf_module)
        pb_estimator = DiscretePBEstimator(env=env, module=pb_module)

        if args.loss in ("DB", "SubTB"):
            # We need a LogStateFlowEstimator

            assert (
                    pf_estimator is not None
            ), f"pf_estimator is None. Command-line arguments: {args}"
            assert (
                    pb_estimator is not None
            ), f"pb_estimator is None. Command-line arguments: {args}"

            if args.tabular:
                module = Tabular(n_states=env.n_states, output_dim=1)
            else:
                module = NeuralNet(
                    input_dim=env.preprocessor.output_dim,
                    output_dim=1,
                    hidden_dim=args.hidden_dim,
                    n_hidden_layers=args.n_hidden,
                    torso=pf_module.torso if args.tied else None,
                )
            logF_estimator = LogStateFlowEstimator(env=env, module=module)

            if args.loss == "DB":
                parametrization = DBParametrization(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                    on_policy=True,
                )
            else:
                parametrization = SubTBParametrization(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                    on_policy=True,
                    weighing=args.subTB_weighing,
                    lamda=args.subTB_lambda,
                )
        elif args.loss == "TB":
            # We need a LogZEstimator
            logZ = LogZEstimator(tensor=torch.tensor(0.0, device=env.device))
            parametrization = TBParametrization(
                pf=pf_estimator,
                pb=pb_estimator,
                logZ=logZ,
                on_policy=True,
            )
        elif args.loss == "ZVar":
            parametrization = LogPartitionVarianceParametrization(
                pf=pf_estimator,
                pb=pb_estimator,
                on_policy=True,
            )

    assert parametrization is not None, f"No parametrization for loss {args.loss}"

    # 3. Create the optimizer
    params = [
        {
            "params": [
                val
                for key, val in parametrization.parameters.items()
                if "logZ" not in key
            ],
            "lr": args.lr,
        }
    ]
    if "logZ.logZ" in parametrization.parameters:
        params.append(
            {
                "params": [parametrization.parameters["logZ.logZ"]],
                "lr": args.lr_Z,
            }
        )

    optimizer = torch.optim.Adam(params)

    # 4. (optional) Create the replay buffer
    replay_buffer = None
    if args.replay_buffer_size > 0:
        replay_buffer = ReplayBuffer(
            env, args.replay_buffer_type, "trajectories", args.replay_buffer_size
        )  # always keep trajectories so that you can compare them for the dist replay buffer name

    visited_terminating_states = env.States.from_batch_shape((0,))
    states_visited = 0
    n_iterations = args.n_trajectories // args.batch_size

    for iteration in trange(n_iterations+1):
        trajectories = parametrization.sample_trajectories(n_samples=args.batch_size)
        states_visited += len(trajectories)
        visited_terminating_states.extend(trajectories.last_states)

        if replay_buffer is not None:
            with torch.no_grad():
                replay_buffer.add(trajectories)
                replay_trajectories = replay_buffer.sample(n_trajectories=args.batch_size)
                trajectories.extend(replay_trajectories)

        training_samples = trajectories_to_training_samples(
            trajectories, parametrization
        )

        optimizer.zero_grad()
        loss = parametrization.loss(training_samples)
        loss.backward()
        optimizer.step()

        to_log = {"loss": loss.item(), "states_visited": states_visited}
        if use_wandb:
            wandb.log(to_log, step=iteration)
        if iteration % args.validation_interval == 0:
            validation_info = validate(
                env,
                parametrization,
                args.validation_samples,
                visited_terminating_states,
            )
            if use_wandb:
                wandb.log(validation_info, step=iteration)
            to_log.update(validation_info)
            tqdm.write(f"{iteration}: {to_log}")

    return to_log


def plot(reward_raw, states, im_show):
    reward = reward_raw.numpy()
    x = states.states_tensor[:, :, 0]
    y = states.states_tensor[:, :, 1]

    def plot2d(reward, x, y):
        # 2D plot
        fig2d, ax = plt.subplots()
        im = ax.imshow(reward, cmap=cm.gist_earth, origin="lower")
        fig2d.colorbar(im, ax=ax)
        fig2d.x_ticks = x
        fig2d.y_ticks = y
        if im_show:
            plt.show()
        return fig2d

    def plot3d(reward, x, y):
        # 3D plot
        fig3d, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        ls = LightSource(270, 45)
        rgb = ls.shade(reward, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(x, y, reward, rstride=1, cstride=1, facecolors=rgb,
                               linewidth=0, antialiased=False, shade=False)
        if im_show:
            plt.show()
        return fig3d

    fig2d = plot2d(reward, x, y)
    fig3d = plot3d(reward, x, y)
    return fig2d, fig3d


def run_train(config, use_wandb, im_show):
    if use_wandb:
        wandb.init(project=config["experiment_name"], name=config["name"])
        wandb.config.update(config)

    to_log = train_hypergrid(config, use_wandb)

    # Plotting
    env = HyperGrid(
        config["ndim"], config["height"], config["reward_type"], config["R0"], config["R1"], config["R2"], config["n_means"], config["cov_scale"],
        config["quantize_bins"], device_str="cpu"
    )
    states = env.build_grid()
    reward = env.reward(states)
    if config["env.ndim"] == 2:
        fig2d, fig3d = plot(reward, states, im_show)
        if use_wandb:
            wandb.log({"2d": wandb.Image(fig2d), "3d": wandb.Image(fig3d)})
            wandb.finish()
    else:
        if use_wandb:
            wandb.finish()
    del fig3d, fig2d


if __name__ == "__main__":
    config = {
        "no_cuda": True,  # Prevent CUDA usage
        "ndim": 2,  # Number of dimensions in the environment
        "height": 64,  # Height of the environment
        "R0": 0.1,  # Environment's R0
        "R1": 0.5,  # Environment's R1
        "R2": 2.0,  # Environment's R2
        "seed": 0,  # Random seed, if 0 then a random seed is used
        "batch_size": 6,  # Batch size, i.e. number of trajectories to sample per training iteration
        "loss": 'FM',  # Loss function to use
        "subTB_weighing": 'geometric_within',  # Weighing scheme for SubTB
        "subTB_lambda": 0.9,  # Lambda parameter for SubTB
        "tabular": False,  # Use a lookup table for F, PF, PB instead of an estimator
        "uniform_pb": False,  # Use a uniform PB
        "tied": False,  # Tie the parameters of PF, PB, and F
        "hidden_dim": 256,  # Hidden dimension of the estimators' neural network modules.
        "n_hidden": 2,  # Number of hidden layers (of size `hidden_dim`) in the estimators neural network modules
        "lr": 0.001,  # Learning rate for the estimators' modules
        "lr_Z": 0.1,  # Specific learning rate for Z (only used for TB loss)
        "n_trajectories": int(16*1000),
        # Total budget of trajectories to train on. Training iterations = n_trajectories // batch_size
        "validation_interval": 100,  # How often (in training steps) to validate the parameterization
        "validation_samples": 200000,  # Number of validation samples to use to evaluate the probability mass function.
        "experiment_name": '',  # Name of the wandb project. If empty, don't use wandb
        "name": 'test',  # Name of the run
        "replay_buffer_size": 2,  # Size of the replay buffer
        "replay_buffer_type": "Dist",  # Type of the replay buffer
        "reward_type": "default",  # Type of reward function
        "n_means": 4,  # Number of means for the GMM reward function
        "quantize_bins": 4,  # Number of quantization bins for the GMM reward function, if -1 no quantization is performed
        "cov_scale": 7.0  # Scale of the covariance matrix for the GMM reward function
    }

    run_train(config, use_wandb=False, im_show=True)
