"""
Copied and Adapted from https://github.com/Tikquuss/GflowNets_Tutorial
"""

from typing import ClassVar, Literal, Tuple, cast

import torch
from einops import rearrange
from gymnasium.spaces import Discrete
from torchtyping import TensorType

from gfn.containers.states import States
from gfn.envs.env import Env
from gfn.envs.preprocessors import (
    IdentityPreprocessor,
    KHotPreprocessor,
    OneHotPreprocessor,
)
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import cm
from matplotlib.colors import LightSource

# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]

preprocessors_dict = {
    "KHot": KHotPreprocessor,
    "OneHot": OneHotPreprocessor,
    "Identity": IdentityPreprocessor,
}


class HyperGrid(Env):
    def __init__(
        self,
        ndim: int = 2,
        height: int = 4,
        R0: float = 0.1,
        R1: float = 0.5,
        R2: float = 2.0,
        reward_name: Literal["cos","gmm-grid","gmm-random","center","corner","default"] = "default",
        num_means:int = 4,
        cov_scale: float = 1.0,
        quantize_bins: int = -1,
        device_str: Literal["cpu", "cuda"] = "cpu",
        preprocessor_name: Literal["KHot", "OneHot", "Identity"] = "KHot",
    ):
        """HyperGrid environment from the GFlowNets paper.
        The states are represented as 1-d tensors of length `ndim` with values in
        {0, 1, ..., height - 1}.
        A preprocessor transforms the states to the input of the neural network,
        which can be a one-hot, a K-hot, or an identity encoding.

        Args:
            ndim (int, optional): dimension of the grid. Defaults to 2.
            height (int, optional): height of the grid. Defaults to 4.
            R0 (float, optional): reward parameter R0. Defaults to 0.1.
            R1 (float, optional): reward parameter R1. Defaults to 0.5.
            R2 (float, optional): reward parameter R1. Defaults to 2.0.
            reward_cos (bool, optional): Which version of the reward to use. Options: ["cos", "gmm","default"].
            device_str (str, optional): "cpu" or "cuda". Defaults to "cpu".
            preprocessor_name (str, optional): "KHot" or "OneHot" or "Identity". Defaults to "KHot".
        """
        self.ndim = ndim
        self.height = height
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.reward_name = reward_name
        self.num_means = num_means
        self.quantize_bins = quantize_bins
        self.cov_scale = cov_scale
        self.offset = cov_scale

        s0 = torch.zeros(ndim, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full(
            (ndim,), fill_value=-1, dtype=torch.long, device=torch.device(device_str)
        )

        action_space = Discrete(ndim + 1)

        if preprocessor_name == "Identity":
            preprocessor = IdentityPreprocessor(output_shape=(ndim,))
        elif preprocessor_name == "KHot":
            preprocessor = KHotPreprocessor(
                height=height, ndim=ndim, get_states_indices=self.get_states_indices
            )
        elif preprocessor_name == "OneHot":
            preprocessor = OneHotPreprocessor(
                n_states=self.n_states,
                get_states_indices=self.get_states_indices,
            )
        else:
            raise ValueError(f"Unknown preprocessor {preprocessor_name}")

        super().__init__(
            action_space=action_space,
            s0=s0,
            sf=sf,
            device_str=device_str,
            preprocessor=preprocessor,
        )

    def make_States_class(self) -> type[States]:
        "Creates a States class for this environment"
        env = self

        class HyperGridStates(States):

            state_shape: ClassVar[tuple[int, ...]] = (env.ndim,)
            s0 = env.s0
            sf = env.sf

            @classmethod
            def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
            ) -> StatesTensor:
                "Creates a batch of random states."
                states_tensor = torch.randint(
                    0, env.height, batch_shape + env.s0.shape, device=env.device
                )
                return states_tensor

            def make_masks(self) -> Tuple[ForwardMasksTensor, BackwardMasksTensor]:
                "Mask illegal (forward and backward) actions."
                forward_masks = torch.ones(
                    (*self.batch_shape, env.n_actions),
                    dtype=torch.bool,
                    device=env.device,
                )
                backward_masks = torch.ones(
                    (*self.batch_shape, env.n_actions - 1),
                    dtype=torch.bool,
                    device=env.device,
                )

                return forward_masks, backward_masks

            def update_masks(self) -> None:
                "Update the masks based on the current states."
                # The following two lines are for typing only.
                self.forward_masks = cast(ForwardMasksTensor, self.forward_masks)
                self.backward_masks = cast(BackwardMasksTensor, self.backward_masks)

                self.forward_masks[..., :-1] = self.states_tensor != env.height - 1
                self.backward_masks = self.states_tensor != 0

        return HyperGridStates

    def is_exit_actions(self, actions: TensorLong) -> TensorBool:
        return actions == self.action_space.n - 1

    def maskless_step(self, states: StatesTensor, actions: TensorLong) -> None:
        states.scatter_(-1, actions.unsqueeze(-1), 1, reduce="add")

    def maskless_backward_step(self, states: StatesTensor, actions: TensorLong) -> None:
        states.scatter_(-1, actions.unsqueeze(-1), -1, reduce="add")

    def not_quantized_reward(self, final_states: States) -> TensorFloat:
        final_states_raw = final_states.states_tensor
        R0, R1, R2 = (self.R0, self.R1, self.R2)
        ax = abs(final_states_raw / (self.height - 1) - 0.5)
        if self.reward_name == "cos":
            pdf_input = ax * 5
            pdf = 1.0 / (2 * torch.pi) ** 0.5 * torch.exp(-(pdf_input**2) / 2)
            reward = R0 + ((torch.cos(ax * 50) + 1) * pdf).prod(-1) * R1
        elif self.reward_name in ["gmm-random","gmm-grid","center","corner"]:
            GMM = self.GMM_generate()
            reward = self.GMM_compute_reward(GMM,final_states_raw)
            reward = torch.tensor(reward, dtype=torch.float32)
            # scale up for more stable training
            reward *= 10
        elif self.reward_name == "default":
            reward = (
                R0 + (0.25 < ax).prod(-1) * R1 + ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2
            )
        else:
            raise ValueError(f"Unknown reward {self.reward_name}")
        return reward

    @property
    def max_reward_value(self) -> TensorFloat:
        return torch.max(self.not_quantized_reward(self.terminating_states)).item()
    @property
    def min_reward_value(self) -> TensorFloat:
        return torch.min(self.not_quantized_reward(self.terminating_states)).item()

    def true_reward(self, final_states: States) -> TensorFloat:
        """
        Function that quantizes the reward values.
        self.quantize_bins: number of values the rewards will be rounded to; if -1 no quantization is performed
        """
        reward = self.not_quantized_reward(final_states)
        if self.quantize_bins == -1 or reward.numel() == 0:
            return reward
        else:
            boundaries = torch.linspace(self.min_reward_value,self.max_reward_value, self.quantize_bins+1)
            indices = torch.bucketize(reward,boundaries,right = True)
            # if element of list is greater than len(boundaries) reduce value by 1
            indices[indices > (len(boundaries)-1)] = len(boundaries)-1
            reward = boundaries[indices]
            return reward


    def log_reward(self, final_states: States) -> TensorFloat:
        return torch.log(self.true_reward(final_states))

    def get_states_indices(self, states: States) -> TensorLong:
        states_raw = states.states_tensor

        canonical_base = self.height ** torch.arange(
            self.ndim - 1, -1, -1, device=states_raw.device
        )
        indices = (canonical_base * states_raw).sum(-1).long()
        return indices

    def get_terminating_states_indices(self, states: States) -> TensorLong:
        return self.get_states_indices(states)

    @property
    def n_states(self) -> int:
        return self.height**self.ndim

    @property
    def n_terminating_states(self) -> int:
        return self.n_states

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        all_states = self.all_states
        assert torch.all(
            self.get_states_indices(all_states)
            == torch.arange(self.n_states, device=self.device)
        )
        true_dist = self.reward(all_states)
        true_dist /= true_dist.sum()
        return true_dist

    @property
    def log_partition(self) -> float:
        grid = self.build_grid()
        rewards = self.reward(grid)
        return rewards.sum().log().item()

    def build_grid(self) -> States:
        "Utility function to build the complete grid"
        H = self.height
        ndim = self.ndim
        grid_shape = (H,) * ndim + (ndim,)  # (H, ..., H, ndim)
        grid = torch.zeros(grid_shape, device=self.device)
        for i in range(ndim):
            grid_i = torch.linspace(start=0, end=H - 1, steps=H)
            for _ in range(i):
                grid_i = grid_i.unsqueeze(1)
            grid[..., i] = grid_i

        rearrange_string = " ".join([f"n{i}" for i in range(1, ndim + 1)])
        rearrange_string += " ndim -> "
        rearrange_string += " ".join([f"n{i}" for i in range(ndim, 0, -1)])
        rearrange_string += " ndim"
        grid = rearrange(grid, rearrange_string).long()
        return self.States(grid)

    @property
    def all_states(self) -> States:
        grid = self.build_grid()
        flat_grid = rearrange(grid.states_tensor, "... ndim -> (...) ndim")
        return self.States(flat_grid)

    @property
    def terminating_states(self) -> States:
        return self.all_states

    # Util functions for GMMs
    def generate_means_on_grid(self):
        """"
        Function that generates means on a grid in multiple dimensions.
        ndim: number of dimensions
        height: height of the grid
        num_means: number of means
        offset: offset of the grid
        """
        num_means_per_dim = int(self.num_means**(self.ndim**-1))
        means = np.linspace(0 + self.offset, self.height-self.offset, num_means_per_dim)
        means = np.meshgrid(*[means for i in range(self.ndim)])
        means = np.array(means).reshape(self.ndim, -1).T
        return means

    def generate_mean_in_center(self):
        """
        Function that generates a mean in the center of the grid.
        ndim: number of dimensions
        """

        mean = np.ones(self.ndim)*self.height/2
        return mean

    def generate_mean_in_corner(self):
        """
        Function that generates a mean in the corner of the grid.
        ndim: number of dimensions
        """
        mean = np.ones(self.ndim)*self.height-self.offset
        return mean

    def generate_means_random(self):
        """
        Function that generates means randomly.
        ndim: number of dimensions
        num_means: number of means per dimension
        height: height of the grid
        offset: offset of the grid
        """
        means = np.random.uniform(0+self.offset, self.height-self.offset, (self.num_means, self.ndim))
        return means

    def GMM_generate(self):
        # Get means and covariance matrices
        if self.reward_name == "gmm-grid":
            means = self.generate_means_on_grid()
            covs = [np.eye(self.ndim)*self.cov_scale] * self.num_means
        elif self.reward_name == "gmm-random":
            means = self.generate_means_random()
            covs = [np.eye(self.ndim)*self.cov_scale] * self.num_means
        elif self.reward_name == "center":
            means = [self.generate_mean_in_center()]
            covs = [np.eye(self.ndim)*self.cov_scale]
        elif self.reward_name == "corner":
            means = [self.generate_mean_in_corner()]
            covs = [np.eye(self.ndim)*self.cov_scale]
        else:
            raise ValueError("mean_strategy not recognized")

        # Create Gaussian mixture model
        GMM = []
        for mean,cov in zip(means,covs):
            GMM.append(multivariate_normal(mean, cov))
        return GMM

    def GMM_compute_reward(self,GMM,state):
        """
        Function that gets the reward of a state given a Gaussian mixture model
        state: state
        GMM: Gaussian mixture model
        """
        reward = 0
        for Gaussian in GMM:
            reward += Gaussian.pdf(state)
        return reward


    # def vary_smoothness(self,reward):
    #     """
    #     Function varies the quantize_bins of the reward function
    #     if quantize_bins == -1: no change to reward
    #     """
    #     if self.quantize_bins == -1:
    #         return reward
    #     else:
    #         reward = np.round(reward*self.quantize_bins,2)/self.quantize_bins
    #         return reward

