from argparse import Namespace

import matplotlib.pyplot as plt
import torch
import wandb
from matplotlib import cm
from matplotlib.colors import LightSource
from tqdm import tqdm, trange

from argparse import ArgumentParser

import torch
import wandb
from tqdm import tqdm, trange

from gfn.containers import ReplayBuffer
from gfn.gflownet import (
    DBGFlowNet,
    FMGFlowNet,
    LogPartitionVarianceGFlowNet,
    ModifiedDBGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
)
from gfn.gym import HyperGrid
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from gfn.utils.common import validate
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

    # 2. Create the gflownets.
    #    For this we need modules and estimators.
    #    Depending on the loss, we may need several estimators:
    #       one (forward only) for FM loss,
    #       two (forward and backward) or other losses
    #       three (same, + logZ) estimators for TB.
    gflownet = None
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
        estimator = DiscretePolicyEstimator(env=env, module=module, forward=True, greedy_eps=args.greedy_eps,temperature=args.temperature, sf_bias=args.sf_bias, epsilon=args.epsilon)
        gflownet = FMGFlowNet(estimator)
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

        pf_estimator = DiscretePolicyEstimator(env=env, module=pf_module, forward=True, greedy_eps=args.greedy_eps,temperature=args.temperature, sf_bias=args.sf_bias, epsilon=args.epsilon)
        pb_estimator = DiscretePolicyEstimator(env=env, module=pb_module, forward=False)

        if args.loss == "ModifiedDB":
            gflownet = ModifiedDBGFlowNet(
                pf_estimator,
                pb_estimator,
                True if args.replay_buffer_size == 0 else False,
            )

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

            logF_estimator = ScalarEstimator(env=env, module=module)
            if args.loss == "DB":
                gflownet = DBGFlowNet(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                    on_policy=True if args.replay_buffer_size == 0 else False,
                )
            else:
                gflownet = SubTBGFlowNet(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                    on_policy=True if args.replay_buffer_size == 0 else False,
                    weighting=args.subTB_weighting,
                    lamda=args.subTB_lambda,
                )
        elif args.loss == "TB":
            gflownet = TBGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
                on_policy=True if args.replay_buffer_size == 0 else False,
            )
        elif args.loss == "ZVar":
            gflownet = LogPartitionVarianceGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
                on_policy=True if args.replay_buffer_size == 0 else False,
            )

    assert gflownet is not None, f"No gflownet for loss {args.loss}"


    # 3. (optional) Create the replay buffer
    replay_buffer = None
    if args.replay_buffer_size > 0:
        replay_buffer = ReplayBuffer(
            env, args.replay_buffer_type, "trajectories", args.replay_buffer_size
        )  # always keep trajectories so that you can compare them for the dist replay buffer name

    # replay_buffer = None
    # if args.replay_buffer_size > 0:
    #     if args.loss in ("TB", "SubTB", "ZVar"):
    #         objects_type = "trajectories"
    #     elif args.loss in ("DB", "ModifiedDB"):
    #         objects_type = "transitions"
    #     elif args.loss == "FM":
    #         objects_type = "states"
    #     else:
    #         raise NotImplementedError(f"Unknown loss: {args.loss}")
    #     replay_buffer = ReplayBuffer(
    #         env, objects_type=objects_type, capacity=args.replay_buffer_size
    #     )

    # 4. Create the optimizer
    # Policy parameters have their own LR.
    params = [
        {
            "params": [
                v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"
            ],
            "lr": args.lr,
        }
    ]

    # Log Z gets dedicated learning rate (typically higher).
    if "logZ" in dict(gflownet.named_parameters()):
        params.append(
            {
                "params": [dict(gflownet.named_parameters())["logZ"]],
                "lr": args.lr_Z,
            }
        )

    optimizer = torch.optim.Adam(params)

    visited_terminating_states = env.States.from_batch_shape((0,))

    states_visited = 0
    n_iterations = args.n_trajectories // args.batch_size
    for iteration in trange(n_iterations+1):
        # trajectories = gflownet.sample_trajectories(n_samples=args.batch_size)
        # training_samples = gflownet.to_training_samples(trajectories)
        # if replay_buffer is not None:
        #     with torch.no_grad():
        #         replay_buffer.add(training_samples)
        #         training_objects = replay_buffer.sample(n_trajectories=args.batch_size)
        # else:
        #     training_objects = training_samples
        #

        trajectories = gflownet.sample_trajectories(n_samples=args.batch_size)
        states_visited += len(trajectories)
        visited_terminating_states.extend(trajectories.last_states)

        if replay_buffer is not None:
            with torch.no_grad():
                replay_buffer.add(trajectories)
                replay_trajectories = replay_buffer.sample(n_trajectories=args.batch_size)
                trajectories.extend(replay_trajectories)

        training_samples = gflownet.to_training_samples(trajectories)

        optimizer.zero_grad()
        loss = gflownet.loss(training_samples)
        loss.backward()
        optimizer.step()

        #to prevent not logging metric is errors
        #to_log = {"loss": loss.item(), "states_visited": states_visited}
        #if use_wandb:
        #    wandb.log(to_log, step=iteration)
        if iteration % args.validation_interval == 0:
            to_log = {"loss": loss.item(), "states_visited": states_visited}
            validation_info = validate(
                env,
                gflownet,
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
    x = states.tensor[:, :, 0]
    y = states.tensor[:, :, 1]

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


def run_train(config, folder="logs", use_wandb=False, im_show=False):
    if use_wandb:
        experiment_name = config["experiment_name"]
        wandb.init(project=f"{folder}_{experiment_name}", name=config["name"])
        wandb.config.update(config)

    to_log = train_hypergrid(config, use_wandb)

    # Plotting
    env = HyperGrid(
        config["ndim"], config["height"], config["reward_type"], config["R0"], config["R1"], config["R2"], config["n_means"], config["cov_scale"],
        config["quantize_bins"], device_str="cpu"
    )
    states = env.build_grid()
    reward = env.reward(states)
    if config["ndim"] == 2:
        fig2d, fig3d = plot(reward, states, im_show)
        if use_wandb:
            wandb.log({"2d": wandb.Image(fig2d), "3d": wandb.Image(fig3d)})
            del fig3d, fig2d
            wandb.finish()
    else:
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    config = {
        "no_cuda": True,  # Prevent CUDA usage
        "ndim": 2,  # Number of dimensions in the environment
        "height": 32,  # Height of the environment
        "R0": 0.1,  # Environment's R0
        "R1": 0.5,  # Environment's R1
        "R2": 2.0,  # Environment's R2
        "seed": 10,  # Random seed, if 0 then a random seed is used
        "batch_size": 16,  # Batch size, i.e. number of trajectories to sample per training iteration
        "loss": 'TB',  # Loss function to use
        "subTB_weighting": 'geometric_within',  # Weighing scheme for SubTB
        "subTB_lambda": 0.9,  # Lambda parameter for SubTB
        "tabular": False,  # Use a lookup table for F, PF, PB instead of an estimator
        "uniform_pb": False,  # Use a uniform PB
        "tied": False,  # Tie the parameters of PF, PB, and F
        "hidden_dim": 256,  # Hidden dimension of the estimators' neural network modules.
        "n_hidden": 2,  # Number of hidden layers (of size `hidden_dim`) in the estimators neural network modules
        "lr": 0.001,  # Learning rate for the estimators' modules
        "lr_Z": 0.1,  # Specific learning rate for Z (only used for TB loss)
        "n_trajectories": int(16*10),
        # Total budget of trajectories to train on. Training iterations = n_trajectories // batch_size
        "validation_interval": 10,  # How often (in training steps) to validate the parameterization
        "validation_samples": 200000,  # Number of validation samples to use to evaluate the probability mass function.
        "experiment_name": '',  # Name of the wandb project. If empty, don't use wandb
        "name": 'test',  # Name of the run
        "replay_buffer_size": 2,  # Size of the replay buffer
        "replay_buffer_type": "Dist",  # Type of the replay buffer
        "reward_type": "GMM-random",  # Type of reward function
        "n_means": 4,  # Number of means for the GMM reward function
        "quantize_bins": 4,  # Number of quantization bins for the GMM reward function, if -1 no quantization is performed
        "cov_scale": 7.0,  # Scale of the covariance matrix for the GMM reward function
        "greedy_eps": 1, # if > 0 , then we go off policy using greedy epsilon exploration
        "temperature": 5.0,  # scalar to divide the logits by before softmax. Does nothing if greedy_eps is 0.
        "sf_bias": 0.0, # scalar to subtract from the exit action logit before dividing by temperature. Does nothing if greedy_eps is 0.
        "epsilon": 0.2, # with probability epsilon, a random action is chosen. Does nothing if greedy_eps is 0.
    }

    run_train(config, folder="logs_debug" , use_wandb=False, im_show=True)
