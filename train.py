
import wandb
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.utils import trajectories_to_training_samples, validate
from scripts.configs import load_config, make_env, make_loss, make_optim, make_sampler
from tqdm import tqdm, trange
import torch
from gfn.containers import Trajectories


def train(config,use_wandb):
    torch.manual_seed(config["seed"])

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
            env, loss_fn, capacity=config["replay_buffer_size"], objects_type= "trajectories"
        ) #always keep trajectories so that you can compare them for the dist replay buffer name

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

        if replay_buffer is not None:
            if config["replay_buffer_name"] == "fifo":
                with torch.no_grad():
                    training_samples = trajectories_to_training_samples(trajectories, loss_fn)
                    replay_buffer.add(training_samples)
            elif config["replay_buffer_name"] == "dist":

                trajectories.extend(replay_buffer.training_objects)
                all_last_states = trajectories.last_states.states_tensor

                #samples_last_states = trajectories.last_states.states_tensor
                #buffer_last_states = replay_buffer.training_objects.last_states.states_tensor
                #all_last_states = torch.cat([samples_last_states, buffer_last_states])

                #buffer = torch.tensor([[0, 0], [1, 1], [2, 2]])
                #new_samples = torch.tensor([[1, 2], [3, 4]])
                # First vmap for every new sample
                batched_func = torch.vmap(lambda x, y: torch.abs(x - y).sum(),
                                          in_dims=(0, None))  # (buffer,new_samples[0])
                # Vmap over all new samples
                distances = torch.vmap(batched_func, in_dims=(None, 0))(all_last_states, all_last_states)
                # Get one number per new sample
                distances = distances.sum(dim=1)
                # k is number of elements to be put in the replay buffer
                k = min(config["replay_buffer_size"],len(distances))
                values,indices = torch.topk(distances, k=k, dim=0)

                # Reinitialize replay buffer
                replay_buffer = ReplayBuffer(
                    env, loss_fn, capacity=config["replay_buffer_size"], objects_type="trajectories"
                )
                replay_buffer.add(trajectories[indices])

                # traj_states = trajectories.states[indices]
                # traj_actions = trajectories.actions[indices]
                # traj_when_is_done = trajectories.when_is_done[indices]
                # traj_is_backward = trajectories.is_backward[indices]
                # traj_log_rewards = trajectories.log_rewards[indices]
                # traj_log_probs = trajectories.log_probs[indices]
                #
                #
                # trajectories = Trajectories(
                #     env=env,
                #     states=traj_states,
                #     actions=traj_actions,
                #     when_is_done=traj_when_is_done,
                #     is_backward=traj_is_backward,
                #     log_rewards=traj_log_rewards,
                #     log_probs=traj_log_probs,
                # )
                #replay_buffer.add(training_samples)
            training_trajectories = replay_buffer.sample(n_trajectories=config["batch_size"])

        else:
            training_trajectories = trajectories
        training_objects = trajectories_to_training_samples(training_trajectories, loss_fn)

        # print(training_objects.actions)
        # print(training_objects.next_states.states_tensor)
        # print(training_objects.rewards)

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
