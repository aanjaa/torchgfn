import functools
import json
import os
import shutil

from ray import air, tune
from ray.tune.search.basic_variant import BasicVariantGenerator

from train import train
from utils import change_config

REWARD_NAMES = ["cos", "gmm-grid", "gmm-random", "center", "default", "corner"]
LOSS_NAMES = ["flowmatching", "detailed-balance", "trajectory-balance", "sub-tb"]
EXPERIMENT_NAMES = ["reward_losses", "smoothness_losses", "searchspaces_losses"]
average_over_multiple_seeds = False
num_samples = 1

CONFIG = {'env': {'device': 'cpu',
                  'ndim': 2,
                  'height': 32,
                  'R0': 0.1,
                  'R1': 0.5,
                  'R2': 2.0,
                  'reward_name': 'gmm-grid',  # ["cos","gmm-grid","gmm-random","center","corner","default"]
                  'num_means': 4,
                  'cov_scale': 7.0,
                  'quantize_bins': -1,
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
          'n_iterations': 1001,  # 1001,
          'replay_buffer_size': 0,
          'no_cuda': False,
          'name': 'debug',
          'experiment_name': 'debug',
          'validation_interval': 100,
          'validation_samples': 10000,  # 200000,
          'resample_for_validation': False}


def run_tune(search_space, num_samples):
    experiment_name = search_space["experiment_name"]
    name = search_space["name"]

    local_dir = os.path.join(os.getcwd(), "logs")
    log_dir = os.path.join(local_dir, experiment_name, name)
    try:
        os.makedirs(log_dir)
    except:
        pass

    metric = "l1_dist"

    # Save the search space by saving this file itself
    shutil.copy(__file__, os.path.join(log_dir + "/ray.py"))
    tuner = tune.Tuner(
        # tune.with_resources(functools.partial(run_fewshot, raytune=True), {"cpu": 1.0, "gpu": 1.0}),
        tune.with_resources(functools.partial(train, use_wandb=False), {"cpu": 1.0, "gpu": 1.0}),
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
        run_config=air.RunConfig(name="details", verbose=1,
                                 local_dir=log_dir)
    )

    results = tuner.fit()

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

    ## EXPERIMENT 1: Which loss for reward function?
    def run_hypergrid_experiment(experiment_name):

        lr = tune.grid_search([0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001])
        if average_over_multiple_seeds:
            seed = tune.grid_search(list(range(3)))
        else:
            seed = 0

        search_spaces = []
        if experiment_name == "reward_losses":
            ndim = 2
            quantize_bins = -1

            for loss_name in LOSS_NAMES:
                for reward_name in REWARD_NAMES:
                    if reward_name == "cos":
                        height = 100
                    else:
                        height = 32

                    name = f"{reward_name}_{loss_name}"

                    search_spaces.append(
                        change_config(CONFIG, name, ndim, height, quantize_bins, reward_name, loss_name, lr, seed,
                                      experiment_name))

        ## EXPERIMENT 2: Which loss for higher ndim and height?
        elif experiment_name == "searchspaces_losses":
            quantize_bins = -1
            reward_name = "default"

            for loss_name in LOSS_NAMES:
                for ndim, height in zip([2, 4], [8, 32]):
                    name = "default_" + f"{ndim}d_{height}h_{loss_name}"
                    search_spaces.append(
                        change_config(CONFIG, name, ndim, height, quantize_bins, reward_name, loss_name, lr, seed,
                                      experiment_name))

        ## EXPERIMENT 3: Which loss for more/less smoothness and height?
        elif experiment_name == "smoothness_losses":
            ndim = 2
            height = 32
            reward_name = "gmm-grid"
            loss_name = "trajectory-balance"
            for quantize_bins in [-1, 4, 10]:
                name = "default_" + f"{quantize_bins}bins"
                search_spaces.append(
                    change_config(CONFIG, name, ndim, height, quantize_bins, reward_name, loss_name, lr, seed,
                                  experiment_name))

        else:
            raise ValueError("Invalid experiment name")

        for search_space in search_spaces:
            try:
                run_tune(search_space, num_samples)
            except:
                print("Error in experiment ", experiment_name, " with search space ", search_space)


    for experiment in EXPERIMENT_NAMES:
        run_hypergrid_experiment(experiment)
