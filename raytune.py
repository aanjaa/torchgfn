import copy
import functools
import json
import os
import shutil

from ray import air, tune
from ray.tune.search.basic_variant import BasicVariantGenerator

from utils import replace_dict_key
from globals import LOSSES, REWARD_TYPES, REPLAY_BUFFER_TYPES
from main_hypergrid import train_hypergrid
from argparse import ArgumentParser
import time

def change_config(config,changes_config):
    for key, value in changes_config.items():
        config = replace_dict_key(config, key, value)
    return config


def run_tune(search_space, num_samples):
    experiment_name = search_space["experiment_name"]
    name = search_space["name"]

    local_dir = os.path.join(os.getcwd(), folder_name)
    log_dir = os.path.join(local_dir, experiment_name, name)
    try:
        os.makedirs(log_dir)
    except:
        pass

    # save the search space
    # get current hour and minute and print them
    with open(os.path.join(log_dir + "/" + time.strftime("%d.%m_%H:%M:%S") + ".json"), 'w') as fp:
        json.dump(search_space, fp, indent=4)

    # Save the search space by saving this file itself
    shutil.copy(__file__, os.path.join(log_dir + "/ray.py"))
    tuner = tune.Tuner(
        #tune.with_resources(functools.partial(train_hypergrid, use_wandb=False), {"cpu": 1.0, "gpu": 1.0}),
        functools.partial(train_hypergrid, use_wandb=False),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode="min",
            num_samples=num_samples,
            # scheduler=tune.schedulers.ASHAScheduler(grace_period=10),
            search_alg=BasicVariantGenerator(constant_grid_search=True),
            # search_alg=OptunaSearch(mode="min", metric="valid_loss_outer"),
            # search_alg=Repeater(OptunaSearch(mode="min", metric="valid_loss_outer"), repeat=2),
        ),
        run_config=air.RunConfig(name="details", verbose=2,local_dir=log_dir, log_to_file=False)
    )

    # Generate txt files
    results = tuner.fit()
    if results.errors:
        print("ERROR!")
    else:
        print("No errors!")
    if results.errors:
        with open(os.path.join(local_dir, "error.txt"), 'w') as file:
            file.write(f"Experiment {experiment_name} failed for {name} with errors {results.errors}")

    with open(os.path.join(log_dir + "/summary.txt"), 'w') as file:
        for i, result in enumerate(results):
            if result.error:
                file.write(f"Trial #{i} had an error: {result.error} \n")
                continue

            file.write(
                f"Trial #{i} finished successfully with a {metric} metric of: {result.metrics[metric]} \n")


    if not average_over_multiple_seeds:
        config = results.get_best_result().config
        with open(os.path.join(log_dir + "/best_config.json"), 'w') as file:
            json.dump(config, file, sort_keys=True, indent=4, skipkeys=True,
                      default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")
    else:
        # Remove all unneeded columns
        df = results.get_dataframe()
        df.to_csv(
            os.path.join(log_dir + "/results.csv"), index=False, float_format='%.8f'
        )
        config_cols = [
            col for col in df.columns
            if ("config/" in col and not any(s in col for s in ["__trial_index__", "seed"]))
        ]
        df = df[[metric] + config_cols]

        # Replace NaNs -> "None", lists -> tuples in config columns, drop nan rows
        df = df.applymap(lambda x: tuple(x) if isinstance(x, list) else x)
        df[config_cols] = df[config_cols].fillna("None")
        df = df.dropna(axis=0)

        # Average over same configs
        df = df.groupby(config_cols)[[metric]].mean().reset_index()
        # Sort by lowest validation loss
        df.sort_values(by=metric, inplace=True)
        # print(df.shape)
        df.to_csv(
            os.path.join(log_dir + "/results_mean.csv"), index=False, float_format='%.8f'
        )

        def _process_dict(dictionary):
            processed_dict = {}

            for key, value in dictionary.items():
                if key.startswith('config/'):
                    key = key[len('config/'):]  # Remove the "config/" prefix from the key

                parts = key.split('/')  # Split the key using the delimiter "/"
                nested_dict = processed_dict

                for part in parts[:-1]:
                    nested_dict.setdefault(part, {})
                    nested_dict = nested_dict[part]

                nested_dict[parts[-1]] = value

            return processed_dict

        # convert df to dict
        df_dict = df.iloc[0].to_dict()
        config = _process_dict(df_dict)

        # save config to json file
        with open(os.path.join(log_dir + "/best_config.json"), 'w') as fp:
            json.dump(config, fp, indent=4)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str,
                        default="searchspaces_losses") #["reward_losses", "smoothness_losses", ["searchspaces_losses"], ["replay_and_capacity"], ["exploration_strategies"]][-3]
    parser.add_argument("--folder", type=str, default="logs_debug")
    args = parser.parse_args()

    folder_name = args.folder
    lr = tune.grid_search([0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001])
    subTB_lambda_grid = tune.grid_search([0.1, 0.3, 0.5, 0.7, 0.9])
    subTB_weighting = 'geometric_within'
    n_iterations = 1000  #1000
    validation_interval = 100  #100

    metric = "KL_forward"  # "l1_dist"

    average_over_multiple_seeds = False
    num_samples = 1

    def run_hypergrid_experiment(experiment_name):
        if average_over_multiple_seeds:
            seed = tune.grid_search(list(range(1,4))) # for 0 seed is randomly chosen
        else:
            seed = 10

        search_spaces = []

        ## EXPERIMENT 1: Which loss for reward function?
        if experiment_name == "reward_losses":
            batch_size = 16
            config = {
                "no_cuda": True,
                "ndim": 2,
                "height": 32,
                "R0": 0.1,
                "R1": 0.5,
                "R2": 2.0,
                "seed": seed,
                "batch_size": batch_size,
                "loss": None, ##
                "subTB_weighting": None,
                "subTB_lambda": None,
                "tabular": False,
                "uniform_pb": False,
                "tied": True,
                "hidden_dim": 256,
                "n_hidden": 2,
                "lr": lr,
                "lr_Z": 0.1,
                "n_trajectories": batch_size * n_iterations,  # Training iterations = n_trajectories // batch_size
                "validation_interval": validation_interval,
                "validation_samples": 10000,
                "experiment_name": experiment_name,
                "name": 'test',
                "replay_buffer_size": 0,
                "replay_buffer_type": None,
                "reward_type": None, ##
                "n_means": 4,
                "quantize_bins": -1,
                "cov_scale": 7.0,
                "greedy_eps": 0,
                "temperature": 1.0,
                "sf_bias": 0.0,
                "epsilon": 0.0,
            }

            #sampler_temperature = 1.0
            #sampler_epsilon = 0.0

            #for reward_name in REWARD_TYPES:
            for reward_name in ["GMM-random"]: #########################TODO:remove!!!!!
                for loss_name in LOSSES:
                    if reward_name == "cos":
                        height = 100
                    else:
                        height = 32

                    name = f"{reward_name}_{loss_name}"
                    if loss_name == "SubTB":
                        changes_config = {
                            "loss": loss_name,
                            "reward_type": reward_name,
                            "name": name,
                            "height": height,
                            "subTB_lambda": subTB_lambda_grid,
                            "subTB_weighting": subTB_weighting,
                        }
                    else:
                        changes_config = {
                            "loss": loss_name,
                            "reward_type": reward_name,
                            "name": name,
                            "height": height,
                        }
                    search_spaces.append(
                        change_config(copy.deepcopy(config), changes_config))


        ## EXPERIMENT 2: Which loss for more/less smoothness and height?
        elif experiment_name == "smoothness_losses":
            batch_size = 16
            config = {
                "no_cuda": True,
                "ndim": 2,
                "height": 32,
                "R0": 0.1,
                "R1": 0.5,
                "R2": 2.0,
                "seed": seed,
                "batch_size": batch_size,
                "loss": None,
                "subTB_weighting": None,
                "subTB_lambda": None,
                "tabular": False,
                "uniform_pb": False,
                "tied": True,
                "hidden_dim": 256,
                "n_hidden": 2,
                "lr": lr,
                "lr_Z": 0.1,
                "n_trajectories": batch_size * n_iterations,  # Training iterations = n_trajectories // batch_size
                "validation_interval": validation_interval,
                "validation_samples": 10000,
                "experiment_name": experiment_name,
                "name": 'test',
                "replay_buffer_size": 0,
                "replay_buffer_type": None,
                "reward_type": "GMM-grid",
                "n_means": 4,
                "quantize_bins": -1,
                "cov_scale": 7.0,
                "greedy_eps": 0,
                "temperature": 1.0,
                "sf_bias": 0.0,
                "epsilon": 0.0,
            }

            for loss in LOSSES:
                for quantize_bins in [-1, 3, 7]:
                    name = f"{loss}_{quantize_bins}bins"

                    if loss == "SubTB":
                        changes_config = {
                            "loss": loss,
                            "quantize_bins": quantize_bins,
                            "name": name,
                            "subTB_lambda": subTB_lambda_grid,
                            "subTB_weighting": subTB_weighting,
                        }
                    else:
                        changes_config = {
                            "loss": loss,
                            "quantize_bins": quantize_bins,
                            "name": name,
                        }
                    search_spaces.append(
                        change_config(copy.deepcopy(config), changes_config))

        ## EXPERIMENT 3: Which loss for higher ndim and height?
        elif experiment_name == "searchspaces_losses":
            batch_size = 16
            config = {
                "no_cuda": True,
                "ndim": None,
                "height": None,
                "R0": 0.1,
                "R1": 0.5,
                "R2": 2.0,
                "seed": seed,
                "batch_size": batch_size,
                "loss": None,  ##
                "subTB_weighting": None,
                "subTB_lambda": None,
                "tabular": False,
                "uniform_pb": False,
                "tied": True,
                "hidden_dim": 256,
                "n_hidden": 2,
                "lr": lr,
                "lr_Z": 0.1,
                "n_trajectories": batch_size * n_iterations,  # Training iterations = n_trajectories // batch_size
                "validation_interval": validation_interval,
                "validation_samples": 10000,
                "experiment_name": experiment_name,
                "name": 'test',
                "replay_buffer_size": 0,
                "replay_buffer_type": None,
                "reward_type": "default",
                "n_means": None,
                "quantize_bins": -1,
                "cov_scale": 7.0,
                "greedy_eps": 0,
                "temperature": 1.0,
                "sf_bias": 0.0,
                "epsilon": 0.0,
            }


            for loss_name in LOSSES:
                #for ndim, height in zip([2, 4], [8, 32]):
                for ndim, height in zip([3, 5], [16, 32]):
                    #n_means = int(2 ** int(ndim))
                    name = "default_" + f"{ndim}d_{height}h_{loss_name}"

                    if loss_name == "SubTB":
                        changes_config = {
                            "loss": loss_name,
                            "ndim": ndim,
                            "height": height,
                            "name": name,
                            "n_means": None,
                            "subTB_lambda": subTB_lambda_grid,
                            "subTB_weighting": subTB_weighting,
                        }
                    else:
                        changes_config = {
                            "loss": loss_name,
                            "ndim": ndim,
                            "height": height,
                            "name": name,
                            "n_means": None,
                        }
                    search_spaces.append(
                        change_config(copy.deepcopy(config), changes_config))


        ## EXPERIMENT 4
        elif experiment_name == "replay_and_capacity":
            # one experiment no replay buffer, one experiment with replay buffer (both strategies: dist and fifo)
            # the other dimension is varying the capacity of the model
            batch_size = 64
            config = {
                "no_cuda": True,
                "ndim": 3,
                "height": 32,
                "R0": 0.1,
                "R1": 0.5,
                "R2": 2.0,
                "seed": seed,
                "batch_size": batch_size,
                "loss": 'TB',
                "subTB_weighting": None,
                "subTB_lambda": None,
                "tabular": False,
                "uniform_pb": False,
                "tied": True,
                "hidden_dim": 256,
                "n_hidden": 2,
                "lr": lr,
                "lr_Z": 0.1,
                "n_trajectories": batch_size * n_iterations,  # 64!!
                "validation_interval": validation_interval,
                "validation_samples": 10000,
                "experiment_name": experiment_name,
                "name": 'test',
                "replay_buffer_size": None,
                "replay_buffer_type": None,
                "reward_type": "default",
                "n_means": None,
                "quantize_bins": -1,
                "cov_scale": 7.0,
                "greedy_eps": 0,
                "temperature": 1.0,
                "sf_bias": 0.0,
                "epsilon": 0.0,
            }


            for n_hidden, hidden_dim in zip([4,2], [256,50]):
                for replay_buffer_size in [500,0]:
                    for replay_buffer_type in REPLAY_BUFFER_TYPES:
                        name = f"nhid{n_hidden}_dimhid{hidden_dim}_replaysize{replay_buffer_size}_replaytype{replay_buffer_type}"
                        changes_config = {
                            "n_hidden": n_hidden,
                            "hidden_dim": hidden_dim,
                            "replay_buffer_size": replay_buffer_size,
                            "replay_buffer_type": replay_buffer_type,
                            "name": name,
                        }
                        search_spaces.append(
                            change_config(copy.deepcopy(config), changes_config))

        elif experiment_name == "exploration_strategies":
            batch_size = 16
            config = {
                "no_cuda": True,
                "ndim": 3,
                "height": 32,
                "R0": 0.0,
                "R1": 0.5,
                "R2": 2.0,
                "seed": seed,
                "batch_size": batch_size,
                "loss": 'TB',
                "subTB_weighting": None,
                "subTB_lambda": None,
                "tabular": False,
                "uniform_pb": False,
                "tied": True,
                "hidden_dim": 256,
                "n_hidden": 2,
                "lr": lr,
                "lr_Z": 0.1,
                "n_trajectories": batch_size * n_iterations,
                "validation_interval": validation_interval,
                "validation_samples": 10000,
                "experiment_name": experiment_name,
                "name": 'test',
                "replay_buffer_size": 0,
                "replay_buffer_type": None,
                "reward_type": "corner",
                "n_means": None,
                "quantize_bins": -1,
                "cov_scale": 7.0,
                "greedy_eps": None,
                "temperature": None,
                "sf_bias": 0.0,
                "epsilon": None,
            }

            for greedy_eps in [0,1]:
                if greedy_eps == 1:
                    for temperature in [1.0, 5.0]:
                        for epsilon in [0.0, 0.1, 0.5]:
                            name = f"exploration{greedy_eps}_eps{epsilon}_temp{temperature}"
                            changes_config = {
                                "greedy_eps": greedy_eps,
                                "temperature": temperature,
                                "epsilon": epsilon,
                                "name": name,
                            }
                            search_spaces.append(
                                change_config(copy.deepcopy(config), changes_config))
                elif greedy_eps == 0:
                    name = f"exploration{greedy_eps}"
                    changes_config = {
                        "greedy_eps": greedy_eps,
                        "name": name,
                    }
                    search_spaces.append(
                        change_config(copy.deepcopy(config), changes_config))
                else:
                    raise ValueError("Invalid greedy_eps")


        else:
            raise ValueError("Invalid experiment name")

        for search_space in search_spaces:
            run_tune(search_space, num_samples)
            # try:
            #     run_tune(search_space, num_samples)
            # except:
            #     print("Error in experiment ", experiment_name, " with search space ", search_space)


    print("Running experiment ", args.experiment_name)
    run_hypergrid_experiment(args.experiment_name)
