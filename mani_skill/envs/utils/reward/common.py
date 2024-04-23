import torch


def tolerance(
    x, lower=0.0, upper=0.0, margin=0.0, sigmoid="gaussian", value_at_margin=0.1
):
    # modified from https://github.com/google-deepmind/dm_control/blob/554ad2753df914372597575505249f22c255979d/dm_control/utils/rewards.py#L93
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
      x: A torch array. (B, 3)
      lower, upper: specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
      margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
          outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
          increasing distance from the nearest bound.
      sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
         'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
      value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`. todo: not implemented yet

    Returns:
      A torch array with values between 0.0 and 1.0.

    Raises:
      ValueError: If `bounds[0] > bounds[1]`.
      ValueError: If `margin` is negative.
    """
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")

    if margin < 0:
        raise ValueError("`margin` must be non-negative.")

    in_bounds = torch.logical_and(lower <= x, x <= upper)

    if margin == 0:
        value = torch.where(in_bounds, torch.tensor(1.0), torch.tensor(0.0))
    else:
        d = torch.where(x < lower, lower - x, x - upper) / margin
        if sigmoid == "gaussian":
            value = torch.where(
                in_bounds, torch.tensor(1.0), torch.exp(-0.5 * (d**2))
            )
        elif sigmoid == "hyperbolic":
            value = torch.where(in_bounds, torch.tensor(1.0), 1 / (1 + torch.exp(d)))
        elif sigmoid == "quadratic":
            value = torch.where(in_bounds, torch.tensor(1.0), 1 - d**2)
        else:
            raise ValueError(f"Unknown sigmoid type {sigmoid!r}.")

    return value
