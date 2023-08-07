"""
Copied and Adapted from https://github.com/Tikquuss/GflowNets_Tutorial
"""
from typing import ClassVar, Literal, Tuple, cast

import torch
from einops import rearrange
from torchtyping import TensorType as TT

from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.gym.helpers.preprocessors import KHotPreprocessor, OneHotPreprocessor
from gfn.preprocessors import EnumPreprocessor, IdentityPreprocessor
from gfn.states import DiscreteStates
from scipy.stats import multivariate_normal
import numpy as np


class HyperGrid(DiscreteEnv):
    def __init__(
        self,
        ndim: int = 2,
        height: int = 4,
        reward_type: Literal["cos", "GMM-grid", "GMM-random", "center", "corner", "default"] = "default",
        R0: float = 0.1,
        R1: float = 0.5,
        R2: float = 2.0,
        n_means: int = 4,
        cov_scale: float = 1.0,
        quantize_bins: int = -1,
        device_str: Literal["cpu", "cuda"] = "cpu",
        preprocessor_name: Literal["KHot", "OneHot", "Identity", "Enum"] = "KHot",
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
            reward_type (bool, optional): Which version of the reward to use
            n_means (int, optional): Number of means for the GMM rewards
            cov_scale (float, optional): Scale of the covariance matrix for the GMM rewards
            quantize_bins (int, optional): Number of bins to quantize reward values
            device_str (str, optional): "cpu" or "cuda". Defaults to "cpu".
            preprocessor_name (str, optional): "KHot" or "OneHot" or "Identity". Defaults to "KHot".
        """
        self.ndim = ndim
        self.height = height
        self.reward_type = reward_type
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.n_means = n_means
        self.cov_scale = cov_scale
        self.quantize_bins = quantize_bins
        self.offset = self.cov_scale/2.0

        s0 = torch.zeros(ndim, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full(
            (ndim,), fill_value=-1, dtype=torch.long, device=torch.device(device_str)
        )

        n_actions = ndim + 1

        if preprocessor_name == "Identity":
            preprocessor = IdentityPreprocessor(output_dim=ndim)
        elif preprocessor_name == "KHot":
            preprocessor = KHotPreprocessor(
                height=height, ndim=ndim, get_states_indices=self.get_states_indices
            )
        elif preprocessor_name == "OneHot":
            preprocessor = OneHotPreprocessor(
                n_states=self.n_states,
                get_states_indices=self.get_states_indices,
            )
        elif preprocessor_name == "Enum":
            preprocessor = EnumPreprocessor(
                get_states_indices=self.get_states_indices,
            )
        else:
            raise ValueError(f"Unknown preprocessor {preprocessor_name}")

        super().__init__(
            n_actions=n_actions,
            s0=s0,
            sf=sf,
            device_str=device_str,
            preprocessor=preprocessor,
        )

    def make_States_class(self) -> type[DiscreteStates]:
        "Creates a States class for this environment"
        env = self

        class HyperGridStates(DiscreteStates):
            state_shape: ClassVar[tuple[int, ...]] = (env.ndim,)
            s0 = env.s0
            sf = env.sf
            n_actions = env.n_actions
            device = env.device

            @classmethod
            def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
            ) -> TT["batch_shape", "state_shape", torch.float]:
                "Creates a batch of random states."
                states_tensor = torch.randint(
                    0, env.height, batch_shape + env.s0.shape, device=env.device
                )
                return states_tensor

            def update_masks(self) -> None:
                "Update the masks based on the current states."
                # The following two lines are for typing only.
                self.forward_masks = cast(
                    TT["batch_shape", "n_actions", torch.bool],
                    self.forward_masks,
                )
                self.backward_masks = cast(
                    TT["batch_shape", "n_actions - 1", torch.bool],
                    self.backward_masks,
                )

                self.forward_masks[..., :-1] = self.tensor != env.height - 1
                self.backward_masks = self.tensor != 0

        return HyperGridStates

    def maskless_step(
        self, states: DiscreteStates, actions: Actions
    ) -> TT["batch_shape", "state_shape", torch.float]:
        new_states_tensor = states.tensor.scatter(-1, actions.tensor, 1, reduce="add")
        return new_states_tensor

    def maskless_backward_step(
        self, states: DiscreteStates, actions: Actions
    ) -> TT["batch_shape", "state_shape", torch.float]:
        new_states_tensor = states.tensor.scatter(-1, actions.tensor, -1, reduce="add")
        return new_states_tensor

    def not_quantized_reward(
        self, final_states: DiscreteStates
    ) -> TT["batch_shape", torch.float]:
        """TODO: Equation governing reward should be placed here."""
        final_states_raw = final_states.tensor
        R0, R1, R2 = (self.R0, self.R1, self.R2)
        ax = abs(final_states_raw / (self.height - 1) - 0.5)
        if self.reward_type == "default":
            reward = (
                R0 + (0.25 < ax).prod(-1) * R1 + ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2
            )
        elif self.reward_type == "cos":
            pdf_input = ax * 5
            pdf = 1.0 / (2 * torch.pi) ** 0.5 * torch.exp(-(pdf_input**2) / 2)
            reward = R0 + ((torch.cos(ax * 50) + 1) * pdf).prod(-1) * R1
        elif self.reward_type in ["GMM-random", "GMM-grid", "center", "corner"]:
            GMMs = self.GMM_generate()
            reward = self.GMM_compute_reward(GMMs, final_states_raw)
            reward = torch.tensor(reward, dtype=torch.float32)
            # scale up for more stable training
            reward *= 10**self.ndim
        else:
            raise ValueError(f"Unknown reward type {self.reward_type}")
        return reward

    @property
    def max_reward_value(self):
        return torch.max(self.not_quantized_reward(self.terminating_states)).item()
    @property
    def min_reward_value(self):
        return torch.min(self.not_quantized_reward(self.terminating_states)).item()

    def true_reward(self, final_states: DiscreteStates) -> TT["batch_shape", torch.float]:
        """
        Function that quantizes the reward values.
        self.quantize_bins: number of values the rewards will be rounded to; if -1 no quantization is performed
        """
        reward = self.not_quantized_reward(final_states)
        if self.quantize_bins == -1 or reward.numel() == 0:
            return reward
        else:
            boundaries = torch.linspace(self.min_reward_value,self.max_reward_value, self.quantize_bins+1, device = self.device)
            indices = torch.bucketize(reward,boundaries,right = True)
            # if element of list is greater than len(boundaries) reduce value by 1
            indices[indices > (len(boundaries)-1)] = len(boundaries)-1
            reward = boundaries[indices]
            return reward

    def log_reward(
        self, final_states: DiscreteStates
    ) -> TT["batch_shape", torch.float]:
        return torch.log(self.true_reward(final_states))

    def get_states_indices(
        self, states: DiscreteStates
    ) -> TT["batch_shape", torch.long]:
        states_raw = states.tensor

        canonical_base = self.height ** torch.arange(
            self.ndim - 1, -1, -1, device=states_raw.device
        )
        indices = (canonical_base * states_raw).sum(-1).long()
        return indices

    def get_terminating_states_indices(
        self, states: DiscreteStates
    ) -> TT["batch_shape", torch.long]:
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

    def build_grid(self) -> DiscreteStates:
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
    def all_states(self) -> DiscreteStates:
        grid = self.build_grid()
        flat_grid = rearrange(grid.tensor, "... ndim -> (...) ndim")
        return self.States(flat_grid)

    @property
    def terminating_states(self) -> DiscreteStates:
        return self.all_states



    # Util functions for GMMs
    def generate_means_on_grid(self):
        """"
        Function that generates means on a grid in multiple dimensions.
        ndim: number of dimensions
        height: height of the grid
        n_means: number of means
        offset: offset of the grid
        """
        n_means_per_dim = int(self.n_means**(self.ndim**-1))
        means = np.linspace(0 + self.offset, self.height-self.offset, n_means_per_dim)
        means = np.meshgrid(*[means for i in range(self.ndim)])
        means = np.array(means).reshape(self.ndim, -1).T
        assert len(means) == self.n_means
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
        n_means: number of means per dimension
        height: height of the grid
        offset: offset of the grid
        """
        np.random.seed(0)
        means = np.random.uniform(0+self.offset, self.height-self.offset, (self.n_means, self.ndim))
        #means = torch.rand((self.n_means, self.ndim)) * (self.height - self.offset - self.offset) + self.offset
        #means = means.numpy()
        return means

    def GMM_generate(self):
        # Get means and covariance matrices
        if self.reward_type == "GMM-grid":
            means = self.generate_means_on_grid()
            covs = [np.eye(self.ndim)*self.cov_scale] * self.n_means
        elif self.reward_type == "GMM-random":
            means = self.generate_means_random()
            covs = [np.eye(self.ndim)*self.cov_scale] * self.n_means
        elif self.reward_type == "center":
            means = [self.generate_mean_in_center()]
            covs = [np.eye(self.ndim)*self.cov_scale]
        elif self.reward_type == "corner":
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

