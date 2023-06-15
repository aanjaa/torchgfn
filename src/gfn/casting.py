# TODO: remove this file eventually
from typing import cast

import torch
from torchtyping import TensorType as TT


def correct_cast(
    forward_masks: TT["batch_shape", "n_actions"] | None,
    backward_masks: TT["batch_shape", "n_actions - 1"] | None,
) -> tuple[
    TT["batch_shape", "n_actions", torch.bool],
    TT["batch_shape", "n_actions - 1", torch.bool],
]:
    """Casts the given masks to the correct type, if they are not None.

    This function is to help with type checking only.

    Args:
        forward_masks: masks for forward actions.
        backward_masks: masks for backward actions.

    Returns: forward_masks and backward_masks cast to bool.
    """
    forward_masks = cast(TT["batch_shape", "n_actions", torch.bool], forward_masks)
    backward_masks = cast(
        TT["batch_shape", "n_actions - 1", torch.bool], backward_masks
    )
    return forward_masks, backward_masks