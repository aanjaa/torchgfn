from dataclasses import dataclass
from typing import Tuple

import torch
from torchtyping import TensorType

from gfn.containers.states import States, correct_cast
from gfn.distributions import EmpiricalTrajectoryDistribution, TrajectoryDistribution
from gfn.envs import Env
from gfn.estimators import LogStateFlowEstimator
from gfn.losses.base import Parametrization, StateDecomposableLoss
from gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler

# Typing
ScoresTensor = TensorType["n_states", float]
LossTensor = TensorType[0, float]


@dataclass
class GGParametrization(Parametrization):
    logF: LogStateFlowEstimator

    def Pi(
        self, env: Env, n_samples: int = 1000, **actions_sampler_kwargs
    ) -> TrajectoryDistribution:
        actions_sampler = DiscreteActionsSampler(self.logF, **actions_sampler_kwargs)
        trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
        trajectories = trajectories_sampler.sample_trajectories(
            n_trajectories=n_samples
        )
        return EmpiricalTrajectoryDistribution(trajectories)


class GGLoss(StateDecomposableLoss):
    def __init__(self, parametrization: GGParametrization) -> None:
        self.parametrization = parametrization
        self.env = parametrization.logF.env
        self.actions_sampler = DiscreteActionsSampler(self.parametrization.logF)

    def __call__(self, states_tuple: Tuple[States, States]) -> LossTensor:
        intermediary_states, terminating_states = states_tuple
        # intermediary_states.extend(terminating_states)
        # terminating_states.extend(intermediary_states)
        logits = self.actions_sampler.get_logits(terminating_states)
        loss = (torch.logsumexp(logits, dim=-1)).pow(2).mean()
        return loss
