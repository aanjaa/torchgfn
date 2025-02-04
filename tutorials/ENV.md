# Creating `torchgfn` environments

To define an environment, the user needs to define the tensor `s0` representing the initial state $s_0$, from which the `state_shape` attribute is inferred, and optionally a tensor representing the sink state $s_f$, which is only used for padding incomplete trajectories. If it is not specified, `sf` is set to a tensor of the same shape as `s0` filled with $-\infty$.

If the environment is discrete, in which case it is an instance of `DiscreteEnv`, the total number of actions should be specified as an attribute.

If the states (as represented in the `States` class) need to be transformed to another format before being processed (by neural networks for example), then the environment should define a `preprocessor` attribute, which should be an instance of the [base preprocessor class](./src/gfn/preprocessors.py). If no preprocessor is defined, the states are used as is (actually transformed using the `IdentityPreprocessor`, which transforms the state tensors to `FloatTensor`s). Implementing a specific preprocessor requires defining the `preprocess` function, and the `output_shape` attribute, which is a tuple representing the shape of *one* preprocessed state.

The user needs to implement the following two abstract functions:
- The method `make_States_class` that creates the corresponding subclass of [`States`](./src/gfn/states.py). For discrete environments, the resulting class should be a subclass of [`DiscreteStates`](./src/gfn/states.py), that implements the `update_masks` method specifying which actions are available at each state.
- The method `make_Actions_class` that creates a subclass of [`Actions`](./src/gfn/actions.py), simply by specifying the required class variables (the shape of an action tensor, the dummy action, and the exit action). This method is implemented by default for all `DiscreteEnv`s.


The logic of the environment is handled by the methods `maskless_step` and `maskless_backward_step`, that need to be implemented, which specify how an action changes a state (going forward and backward). These functions do not need to handle masking for discrete environments, nor checking whether actions are allowed, nor checking whether a state is the sink state, etc... These checks are handled in `Env.step` and `Env.backward_step` functions, that do not need to be implemented. Non discrete environments need to implement the `is_action_valid` function however, taking a batch of states and actions, and returning `True` only if all actions can be taken at the given states.
- The `log_reward` function that assigns the logarithm of a nonnegative reward to every terminating state (i.e. state with all $s_f$ as a child in the DAG). If `log_reward` is not implemented, `reward` needs to be.


For `DiscreteEnv`s, the user can define a`get_states_indices` method that assigns a unique integer number to each state, and a `n_states` property that returns an integer representing the number of states (excluding $s_f$) in the environment. The function `get_terminating_states_indices` can also be implemented and serves the purpose of uniquely identifying terminating states of the environment, which is useful for [tabular `GFNModule`s](./src/gfn/estimators.py). Other properties and functions can be implemented as well, such as the `log_partition` or the `true_dist_pmf` properties.

For reference, it might be useful to look at one of the following provided environments:
- [HyperGrid](./src/gfn/gym/hypergrid.py) is an example of a discrete environment where all states are terminating states. - [DiscreteEBM](./src/gfn/gym/discrete_ebm.py) is an example of a discrete environment where all trajectories are of the same length but only some states are terminating.
- [Box](./src/gfn/gym/box.py) is an example of a continuous environment with a specific `is_action_valid` function.