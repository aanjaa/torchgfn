def replace_dict_key(dictionary, key, value):
    keys = key.split(".")
    current_dict = dictionary

    for k in keys[:-1]:
        if k in current_dict:
            current_dict = current_dict[k]
        else:
            raise KeyError(f"Key '{key}' does not exist in the dictionary.")

    last_key = keys[-1]
    if last_key in current_dict:
        current_dict[last_key] = value
    else:
        raise KeyError(f"Key '{key}' does not exist in the dictionary.")

    return dictionary


def change_config(config,name, ndim, height, quantize_bins, reward_name, loss_name, lr, seed, experiment_name):
    changes = {
        "name": name,
        "env.ndim": ndim,
        "env.height": height,
        "env.quantize_bins": quantize_bins,
        "env.reward_name": reward_name,
        "loss.name": loss_name,
        "optim.lr": lr,
        "seed": seed,
        "experiment_name": experiment_name,
    }

    for key, value in changes.items():
        config = replace_dict_key(config, key, value)
    return config