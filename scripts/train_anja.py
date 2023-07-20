import json
from argparse import ArgumentParser

import torch
import wandb
from configs import load_config, make_env, make_loss, make_optim, make_sampler
from tqdm import tqdm, trange

from gfn.containers.replay_buffer import ReplayBuffer
from gfn.utils import trajectories_to_training_samples, validate

if __name__ == "__main__":
    config = {'env': {'device': 'cpu',
             #'ndim': 2,
             #'height': 8,
             'R0': 0.1,
             'R1': 0.5,
             'R2': 2.0,
             'reward_cos': False,
             'preprocessor_name': 'KHot',
             'name': 'hypergrid'},
     'loss': {'module_name': 'NeuralNet',
              'n_hidden_layers': 2,
              'hidden_dim': 256,
              'activation_fn': 'relu',
              'forward_looking': False,
              #'name': 'detailed-balance'},
              },
     'optim': {'lr': 0.001, 'lr_Z': 0.1, 'betas': [0.9, 0.999], 'name': 'adam'},
     'sampler': {'temperature': 1.0, 'sf_bias': 0.0, 'epsilon': 0.0, 'name': None},
     #'seed': 0,
     'batch_size': 16,
     'n_iterations': 1000,
     'replay_buffer_size': 0,
     'no_cuda': False,
     'wandb': '',#'gflownets',
     'validation_interval': 100,
     'validation_samples': 200000,
     'resample_for_validation': False}


    if config["no_cuda"]:
        device_str = "cpu"
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    config["device"] = device_str

    for ndim in [3]:
        config["env"]["ndim"] = ndim

        for height in [32,8]:
            config["env"]["height"] = height

            for training_strategy in ["flowmatching","detailed-balance","trajectory-balance","sub-tb"]:
                config["loss"]["name"] = training_strategy

                for seed in [0, 1, 2]:
                    config["seed"] = seed
                    torch.manual_seed(config["seed"])

                    print("Config:")
                    print(json.dumps(config, indent=2))

                    env = make_env(config)
                    parametrization, loss_fn = make_loss(config, env)
                    optimizer = make_optim(config, parametrization)
                    trajectories_sampler, on_policy = make_sampler(config, env, parametrization)
                    loss_fn.on_policy = on_policy

                    use_replay_buffer = False
                    replay_buffer = None
                    if config["replay_buffer_size"] > 0:
                        use_replay_buffer = True
                        replay_buffer = ReplayBuffer(
                            env, loss_fn, capacity=config["replay_buffer_size"]
                        )

                    use_wandb = len(config["wandb"]) > 0
                    if use_wandb:
                        wandb_name = f"ndim{ndim}_h{height}_{training_strategy}"
                        wandb.init(project=config["wandb"],name=wandb_name)
                        wandb.config.update(config)

                    visited_terminating_states = (
                        env.States.from_batch_shape((0,))
                        if not config["resample_for_validation"]
                        else None
                    )

                    states_visited = 0
                    for i in trange(config["n_iterations"]):
                        trajectories = trajectories_sampler.sample(n_trajectories=config["batch_size"])
                        #print(trajectories.max_length)
                        #print(trajectories.last_states.states_tensor)
                        training_samples = trajectories_to_training_samples(trajectories, loss_fn)
                        if replay_buffer is not None:
                            replay_buffer.add(training_samples)
                            training_objects = replay_buffer.sample(n_trajectories=config["batch_size"])
                        else:
                            training_objects = training_samples

                        optimizer.zero_grad()
                        loss = loss_fn(training_objects)
                        loss.backward()

                        optimizer.step()
                        if visited_terminating_states is not None:
                            visited_terminating_states.extend(trajectories.last_states)

                        states_visited += len(trajectories)
                        to_log = {"loss": loss.item(), "states_visited": states_visited}
                        if use_wandb:
                            wandb.log(to_log, step=i)
                        if i % config["validation_interval"] == 0:
                            validation_info = validate(
                                env,
                                parametrization,
                                config["validation_samples"],
                                visited_terminating_states,
                            )
                            if use_wandb:
                                wandb.log(validation_info, step=i)
                            to_log.update(validation_info)
                            tqdm.write(f"{i}: {to_log}")

                    wandb.finish()
