import copy
import functools
import json
import os
import shutil

from ray import air, tune
from ray.tune.search.basic_variant import BasicVariantGenerator

from utils import replace_dict_key
from globals import LOSSES, REWARD_TYPES, REPLAY_BUFFER_TYPES

average_over_multiple_seeds = False
num_samples = 1

from main_hypergrid import train_hypergrid


def change_config(config,changes_config):
    for key, value in changes_config.items():
        config = replace_dict_key(config, key, value)
    return config


def run_tune(search_space, num_samples):
    experiment_name = search_space["experiment_name"]
    name = search_space["name"]

    local_dir = os.path.join(os.getcwd(), FOLDER_NAME)
    log_dir = os.path.join(local_dir, experiment_name, name)
    try:
        os.makedirs(log_dir)
    except:
        pass

    metric = "l1_dist"

    # Save the search space by saving this file itself
    shutil.copy(__file__, os.path.join(log_dir + "/ray.py"))
    tuner = tune.Tuner(
        tune.with_resources(functools.partial(train_hypergrid, use_wandb=False), {"cpu": 1.0, "gpu": 1.0}),
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
        run_config=air.RunConfig(name="details", verbose=1,local_dir=log_dir, log_to_file=False)
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

    FOLDER_NAME = "logs_missing"
    EXPERIMENTS = ["replay_and_capacity"]

    ## EXPERIMENT 1: Which loss for reward function?
    def run_hypergrid_experiment(experiment_name):

        lr = tune.grid_search([0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001])
        if average_over_multiple_seeds:
            seed = tune.grid_search(list(range(1,4))) # for 0 seed is randomly chosen
        else:
            seed = 10

        search_spaces = []

        if experiment_name == "reward_losses":

            config = {
                "no_cuda": False,
                "ndim": 2,
                "height": None,
                "R0": 0.1,
                "R1": 0.5,
                "R2": 2.0,
                "seed": seed,
                "batch_size": 16,
                "loss": None, ##
                "subTB_weighing": 'geometric_within',
                "subTB_lambda": 0.9,
                "tabular": False,
                "uniform_pb": False,
                "tied": True,
                "hidden_dim": 256,
                "n_hidden": 2,
                "lr": lr,
                "lr_Z": 0.1,
                "n_trajectories": int(16016),  # Training iterations = n_trajectories // batch_size
                "validation_interval": 100,
                "validation_samples": 10000,
                "experiment_name": experiment_name,
                "name": 'test',
                "replay_buffer_size": 0,
                "replay_buffer_type": None,
                "reward_type": None, ##
                "n_means": 4,
                "quantize_bins": -1,
                "cov_scale": 7.0
            }

            #sampler_temperature = 1.0
            #sampler_epsilon = 0.0

            for reward_name in REWARD_TYPES:
                for loss_name in LOSSES:
                    if reward_name == "cos":
                        height = 100
                    else:
                        height = 32

                    name = f"{reward_name}_{loss_name}"

                    changes_config = {
                        "loss": loss_name,
                        "reward_type": reward_name,
                        "name": name,
                        "height": height
                    }
                    search_spaces.append(
                        change_config(copy.deepcopy(config), changes_config))

        ## EXPERIMENT 2: Which loss for higher ndim and height?
        elif experiment_name == "searchspaces_losses":

            config = {
                "no_cuda": True,
                "ndim": None,
                "height": None,
                "R0": 0.1,
                "R1": 0.5,
                "R2": 2.0,
                "seed": seed,
                "batch_size": 16,
                "loss": None, ##
                "subTB_weighing": 'geometric_within',
                "subTB_lambda": 0.9,
                "tabular": False,
                "uniform_pb": False,
                "tied": True,
                "hidden_dim": 256,
                "n_hidden": 2,
                "lr": lr,
                "lr_Z": 0.1,
                "n_trajectories": int(16016),  # Training iterations = n_trajectories // batch_size
                "validation_interval": 100,
                "validation_samples": 10000,
                "experiment_name": experiment_name,
                "name": 'test',
                "replay_buffer_size": 0,
                "replay_buffer_type": None,
                "reward_type": "default",
                "n_means": None,
                "quantize_bins": -1,
                "cov_scale": 7.0
            }

            #sampler_temperature = 1.0
            #sampler_epsilon = 0.0

            for loss_name in LOSSES:
                for ndim, height in zip([2, 4], [8, 32]):

                    n_means = int(2**int(ndim))
                    name = "default_" + f"{ndim}d_{height}h_{loss_name}"

                    changes_config = {
                        "loss": loss_name,
                        "ndim": ndim,
                        "height": height,
                        "name": name,
                        "n_means": n_means,
                    }
                    search_spaces.append(
                        change_config(copy.deepcopy(config),changes_config))


        ## EXPERIMENT 3: Which loss for more/less smoothness and height?
        elif experiment_name == "smoothness_losses":

            config = {
                "no_cuda": True,
                "ndim": 2,
                "height": 32,
                "R0": 0.1,
                "R1": 0.5,
                "R2": 2.0,
                "seed": seed,
                "batch_size": 16,
                "loss": 'TB',
                "subTB_weighing": 'geometric_within',
                "subTB_lambda": 0.9,
                "tabular": False,
                "uniform_pb": False,
                "tied": True,
                "hidden_dim": 256,
                "n_hidden": 2,
                "lr": lr,
                "lr_Z": 0.1,
                "n_trajectories": int(16016),  # Training iterations = n_trajectories // batch_size
                "validation_interval": 100,
                "validation_samples": 10000,
                "experiment_name": experiment_name,
                "name": 'test',
                "replay_buffer_size": 0,
                "replay_buffer_type": None,
                "reward_type": "GMM-grid",
                "n_means": 4,
                "quantize_bins": -1,
                "cov_scale": 7.0
            }

            #sampler_temperature = 1.0
            #sampler_epsilon = 0.0


            for quantize_bins in [-1, 4, 10]:
                name = "default_" + f"{quantize_bins}bins"
                changes_config = {
                    "quantize_bins": quantize_bins,
                    "name": name,
                }
                search_spaces.append(
                    change_config(copy.deepcopy(config), changes_config))

        ## EXPERIMENT 4
        elif experiment_name == "replay_and_capacity":
            # one experiment no replay buffer, one experiment with replay buffer (both strategies: dist and fifo)
            # the other dimension is varying the capacity of the model

            config = {
                "no_cuda": True,
                "ndim": 2,
                "height": 32,
                "R0": 0.1,
                "R1": 0.5,
                "R2": 2.0,
                "seed": seed,
                "batch_size": 64,
                "loss": 'TB',
                "subTB_weighing": 'geometric_within',
                "subTB_lambda": 0.9,
                "tabular": False,
                "uniform_pb": False,
                "tied": True,
                "hidden_dim": 256,
                "n_hidden": 2,
                "lr": lr,
                "lr_Z": 0.1,
                "n_trajectories": int(16016),  # Training iterations = n_trajectories // batch_size
                "validation_interval": 100,
                "validation_samples": 10000,
                "experiment_name": experiment_name,
                "name": 'test',
                "replay_buffer_size": 0,
                "replay_buffer_type": None,
                "reward_type": "GMM-grid",
                "n_means": 4,
                "quantize_bins": -1,
                "cov_scale": 7.0
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

        # elif experiment_name == "exploration_strategies":
        #     ndim = 2
        #     height = 32
        #     quantize_bins = -1
        #     reward_name = "gmm-grid"
        #     loss_name = "trajectory-balance"
        #     replay_buffer_size = 0
        #     replay_buffer_name = ""
        #     n_hidden = 2
        #     hidden_dim = 256
        #     batch_size = 16
        #     num_means = 4
        #
        #     for sampler_temperature in [1.0, 10.0]:
        #         for sampler_epsilon in [0.0, 0.1, 0.5]:
        #             name = f"temp{sampler_temperature}_eps{sampler_epsilon}"
        #             search_spaces.append(
        #                 change_config(name, ndim, height, quantize_bins, reward_name, loss_name, lr, seed,
        #                               experiment_name, replay_buffer_size, replay_buffer_name, n_hidden,
        #                               hidden_dim, sampler_temperature, sampler_epsilon, batch_size, num_means))

        else:
            raise ValueError("Invalid experiment name")

        for search_space in search_spaces:
            run_tune(search_space, num_samples)
            # try:
            #     run_tune(search_space, num_samples)
            # except:
            #     print("Error in experiment ", experiment_name, " with search space ", search_space)


    for experiment in EXPERIMENTS:
        print("Running experiment ", experiment)
        run_hypergrid_experiment(experiment)
