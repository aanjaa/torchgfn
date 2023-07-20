import inspect
from dataclasses import dataclass
from typing import Literal

from gfn.envs import DiscreteEBMEnv, Env, HyperGrid


@dataclass
class HyperGridConfig:
    ndim: int = 2
    height: int = 8
    R0: float = 0.1
    R1: float = 0.5
    R2: float = 2.0
    reward_name: Literal["cos","gmm-grid","gmm-random","center","corner","default"] = "default"
    num_means: int = 4
    cov_scale: float = 7.0
    quantize_bins: int = -1
    preprocessor_name: Literal["KHot", "OneHot", "Identity"] = "KHot"

    def parse(self, device_str: Literal["cpu", "cuda"]) -> Env:
        return HyperGrid(
            ndim=self.ndim,
            height=self.height,
            R0=self.R0,
            R1=self.R1,
            R2=self.R2,
            reward_name=self.reward_name,
            num_means=self.num_means,
            cov_scale=self.cov_scale,
            quantize_bins=self.quantize_bins,
            device_str=device_str,
            preprocessor_name=self.preprocessor_name,
        )


@dataclass
class DiscreteEBMConfig:
    ndim: int = 4
    alpha: float = 1.0
    # You can define your own custom energy function, and pass it to the DiscreteEBMEnv constructor.

    def parse(self, device_str: Literal["cpu", "cuda"]) -> Env:
        return DiscreteEBMEnv(
            ndim=self.ndim,
            alpha=self.alpha,
            device_str=device_str,
        )


def make_env(config: dict) -> Env:
    assert config["env"]["device"] in [
        "cpu",
        "cuda",
    ], "Invalid device: {}. Must be 'cpu' or 'cuda'".format(config["env"]["device"])

    name = config["env"]["name"]
    device = config["env"]["device"]
    if name.lower() == "hypergrid".lower():
        env_class = HyperGridConfig
    elif name.lower() == "discrete-ebm".lower():
        env_class = DiscreteEBMConfig
    else:
        raise ValueError("Invalid env name: {}".format(name))

    args = inspect.getfullargspec(env_class.__init__).args
    env_config = {k: v for k, v in config["env"].items() if k in args}
    return env_class(**env_config).parse(device)
