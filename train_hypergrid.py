"""
The goal of this script is to reproduce some of the published results on the HyperGrid
environment. Run one of the following commands to reproduce some of the results in
[Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259)

python train_hypergrid.py --ndim 4 --height 8 --R0 {0.1, 0.01, 0.001} --tied {--uniform} --loss {TB, DB}
python train_hypergrid.py --ndim 2 --height 64 --R0 {0.1, 0.01, 0.001} --tied {--uniform} --loss {TB, DB}

And run one of the following to reproduce some of the results in
[Learning GFlowNets from partial episodes for improved convergence and stability](https://arxiv.org/abs/2209.12782)
python train_hypergrid.py --ndim {2, 4} --height 12 --R0 {1e-3, 1e-4} --tied --loss {TB, DB, SubTB}
"""

from argparse import ArgumentParser, Namespace

import torch
from tqdm import tqdm, trange

import wandb
from gfn.gym import HyperGrid
from gfn.estimators import LogEdgeFlowEstimator, LogStateFlowEstimator, LogZEstimator
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
from gfn.containers.replay_buffer import ReplayBuffer


def train_hypergrid(args):
    seed = args.seed if args.seed != 0 else torch.randint(int(10e10), (1,))[0].item()
    torch.manual_seed(seed)

    device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    use_wandb = len(args.wandb_project) > 0
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.name, config=args)
        #wandb.config.update(args)

    # 1. Create the environment
    env = HyperGrid(
        args.ndim, args.height, args.R0, args.R1, args.R2, device_str=device_str
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
        ) #always keep trajectories so that you can compare them for the dist replay buffer name

    visited_terminating_states = env.States.from_batch_shape((0,))
    states_visited = 0
    n_iterations = args.n_trajectories // args.batch_size

    for iteration in trange(n_iterations):
        trajectories = parametrization.sample_trajectories(n_samples=args.batch_size)
        states_visited += len(trajectories)
        visited_terminating_states.extend(trajectories.last_states)

        if replay_buffer is not None:
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


if __name__ == "__main__":
    config = {
        "no_cuda": False,  # Prevent CUDA usage
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
        "n_trajectories": int(1e6),  # Total budget of trajectories to train on. Training iterations = n_trajectories // batch_size
        "validation_interval": 100,  # How often (in training steps) to validate the parameterization
        "validation_samples": 200000,  # Number of validation samples to use to evaluate the probability mass function.
        "wandb_project": '',  # Name of the wandb project. If empty, don't use wandb
        "name": 'test',  # Name of the run
        "replay_buffer_size": 2,  # Size of the replay buffer
        "replay_buffer_type": "Dist",  # Type of the replay buffer
        "reward_type": "GMM-grid",  # Type of reward function
        "n_means": 4,  # Number of means for the GMM reward function
        "n_bins": 4,  # Number of quantization bins for the GMM reward function, if -1 no quantization is performed
    }

    args = Namespace(**config)
    to_log = train_hypergrid(args)

