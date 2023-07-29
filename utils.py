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

def convert_str_to_bool(args_obj, args_to_convert):
    for arg in args_to_convert:
        # check format
        curr_value = getattr(args_obj, arg)
        try:
            assert curr_value in ['true', 'false']
        except:
            raise ValueError(f' Argument {arg}={curr_value} needs to provided as true or false')
        # set bool format
        bool_value = True if curr_value == 'true' else False
        setattr(args_obj, arg, bool_value)
    return args_obj
