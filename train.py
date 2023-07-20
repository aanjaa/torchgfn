
import wandb
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.utils import trajectories_to_training_samples, validate
from scripts.configs import load_config, make_env, make_loss, make_optim, make_sampler
from tqdm import tqdm, trange
import torch

def train(config,use_wandb):
    if config["no_cuda"]:
        device_str = "cpu"
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    config["device"] = device_str

    env = make_env(config)
    parametrization, loss_fn = make_loss(config, env)
    optimizer = make_optim(config, parametrization)
    trajectories_sampler, on_policy = make_sampler(config, env, parametrization)
    loss_fn.on_policy = on_policy

    replay_buffer = None
    if config["replay_buffer_size"] > 0:
        replay_buffer = ReplayBuffer(
            env, loss_fn, capacity=config["replay_buffer_size"]
        )

    if use_wandb:
        wandb.init(project=config["experiment_name"], name=config["name"])
        wandb.config.update(config)

    visited_terminating_states = (
        env.States.from_batch_shape((0,))
        if not config["resample_for_validation"]
        else None
    )

    states_visited = 0
    for i in trange(config["n_iterations"]):
        trajectories = trajectories_sampler.sample(n_trajectories=config["batch_size"])
        # print(trajectories.max_length)
        # print(trajectories.last_states.states_tensor)
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

    return to_log