from typing import List, Optional

import torch
from torchtyping import TensorType

from gfn.containers import States, Trajectories
from gfn.envs import Env
from gfn.samplers import ActionSampler, BackwardsActionSampler, FixedActions

# Typing
StatesTensor = TensorType["n_trajectories", "state_shape", torch.float]
ActionsTensor = TensorType["n_trajectories", torch.long]
DonesTensor = TensorType["n_trajectories", torch.bool]


class TrajectoriesSampler:
    def __init__(self, env: Env, action_sampler: ActionSampler):
        self.env = env
        self.action_sampler = action_sampler
        self.is_backwards = isinstance(action_sampler, BackwardsActionSampler)

    def sample_trajectories(
        self, states: Optional[States] = None, n_trajectories: Optional[int] = None
    ) -> Trajectories:
        if states is None:
            assert (
                n_trajectories is not None
            ), "Either states or n_trajectories should be specified"
            states = self.env.reset(batch_shape=(n_trajectories,))
        else:
            assert (
                len(states.batch_shape) == 1
            ), "States should be a linear batch of states"
            n_trajectories = states.batch_shape[0]
        assert states is not None

        dones = states.is_initial_state if self.is_backwards else states.is_sink_state

        trajectories_states: List[StatesTensor] = [states.states]
        trajectories_actions: List[ActionsTensor] = []
        trajectories_dones = (-1) * torch.ones(n_trajectories, dtype=torch.long)
        step = 0

        while not all(dones):
            _, actions = self.action_sampler.sample(states)
            trajectories_actions += [actions]

            if self.is_backwards:
                new_states = self.env.backward_step(states, actions)
            else:
                # TODO: fix this when states is passed as argument
                new_states = self.env.step(states, actions)
            step += 1

            new_dones = (
                new_states.is_initial_state
                if self.is_backwards
                else new_states.is_sink_state
            )
            trajectories_dones[new_dones & ~dones] = step
            states = new_states
            dones = new_dones

            trajectories_states += [states.states]

            if (
                isinstance(self.action_sampler, FixedActions)
                and self.action_sampler.step >= self.action_sampler.total_steps
            ):
                dones = [True]

        trajectories_states = torch.stack(trajectories_states, dim=0)
        trajectories_states = self.env.States(states=trajectories_states)
        trajectories_actions = torch.stack(trajectories_actions, dim=0)

        if self.is_backwards:
            trajectories_rewards = None
        else:
            trajectories_rewards = self.env.reward(states)

        trajectories = Trajectories(
            env=self.env,
            n_trajectories=n_trajectories,
            states=trajectories_states,
            actions=trajectories_actions,
            when_is_done=trajectories_dones,
            rewards=trajectories_rewards,
            last_states=states,
        )

        return trajectories


if __name__ == "__main__":
    from gfn.envs import HyperGrid
    from gfn.estimators import LogitPBEstimator, LogitPFEstimator
    from gfn.models import NeuralNet
    from gfn.preprocessors import (
        IdentityPreprocessor,
        KHotPreprocessor,
        OneHotPreprocessor,
    )
    from gfn.samplers.action_samplers import (
        LogitPBActionSampler,
        LogitPFActionSampler,
        UniformActionSampler,
        UniformBackwardsActionSampler,
    )

    env = HyperGrid(ndim=2, height=4)

    print("---Trying Forward sampling of trajectories---")

    print("Trying the Uniform Action Sample with sf_temperature")
    action_sampler = UniformActionSampler(sf_temperature=2.0)
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=5)
    print(trajectories)

    print("Trying the Fixed Actions Sampler: ")
    action_sampler = FixedActions(
        torch.tensor(
            [[0, 1, 2, 0], [1, 1, 1, 2], [2, 2, 2, 2], [1, 0, 1, 2], [1, 0, 2, 1]]
        )
    )
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=5)
    print(trajectories)

    print("Trying the LogitPFActionSampler: ")
    preprocessors = [
        IdentityPreprocessor(env=env),
        OneHotPreprocessor(env=env),
        KHotPreprocessor(env=env),
    ]
    modules = [
        NeuralNet(
            input_dim=preprocessor.output_dim, hidden_dim=12, output_dim=env.n_actions
        )
        for preprocessor in preprocessors
    ]
    logit_pf_estimators = [
        LogitPFEstimator(preprocessor=preprocessor, env=env, module=module)
        for (preprocessor, module) in zip(preprocessors, modules)
    ]

    logit_pf_action_samplers = [
        LogitPFActionSampler(logit_PF=logit_pf_estimator)
        for logit_pf_estimator in logit_pf_estimators
    ]

    trajectories_samplers = [
        TrajectoriesSampler(env, logit_pf_action_sampler)
        for logit_pf_action_sampler in logit_pf_action_samplers
    ]

    for i, trajectories_sampler in enumerate(trajectories_samplers):
        print(
            i,
            ": Trying the LogitPFActionSampler with preprocessor {}".format(
                preprocessors[i]
            ),
        )
        trajectories = trajectories_sampler.sample_trajectories()
        print(trajectories)
    assert False

    print("---Trying Backwards sampling of trajectories---")
    states = env.StatesBatch(random=True)
    states.states[0] = torch.zeros(2)
    states.update_masks()
    states.update_the_dones()
    print(
        "Trying the Uniform Backwards Action Sampler with one of the initial states being s_0"
    )
    action_sampler = UniformBackwardsActionSampler()
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories(states)
    print(trajectories)

    modules = [
        NeuralNet(
            input_dim=preprocessor.output_dim,
            hidden_dim=12,
            output_dim=env.n_actions - 1,
        )
        for preprocessor in preprocessors
    ]
    logit_pb_estimators = [
        LogitPBEstimator(preprocessor=preprocessor, env=env, module=module)
        for (preprocessor, module) in zip(preprocessors, modules)
    ]

    logit_pb_action_samplers = [
        LogitPBActionSampler(logit_PB=logit_pb_estimator)
        for logit_pb_estimator in logit_pb_estimators
    ]

    trajectories_samplers = [
        TrajectoriesSampler(env, logit_pb_action_sampler)
        for logit_pb_action_sampler in logit_pb_action_samplers
    ]

    for i, trajectories_sampler in enumerate(trajectories_samplers):
        print(
            i,
            ": Trying the LogitPBActionSampler with preprocessor {}".format(
                preprocessors[i]
            ),
        )
        states = env.StatesBatch(random=True)
        states.update_masks()
        states.update_the_dones()
        trajectories = trajectories_samplers[i].sample_trajectories(states)
        print(trajectories)

    print("---Making Sure Last states are computed correctly---")
    action_sampler = UniformActionSampler(sf_temperature=2.0)
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories()
    if trajectories.last_states.states.equal(trajectories.get_last_states_raw()):
        print("Last states are computed correctly")
    else:
        print("WARNING !! Last states are not computed correctly")

    print("---Testing SubSampling---")
    print(
        f"There are {trajectories.n_trajectories} original trajectories, from which we will sample 2"
    )
    print(trajectories)
    sampled_trajectories = trajectories.sample(n_trajectories=2)
    print("The two sampled trajectories are:")
    print(sampled_trajectories)
