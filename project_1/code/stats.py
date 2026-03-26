from typing import Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray
from numba import njit


class BootstrapResult(NamedTuple):
  statistic: float
  bias: float
  standard_error: float


@njit
def seed_numba(s: int):
  """Numba helper function to seed the random number generator."""
  np.random.seed(s)


def timeseries_bootstrap(
  data: NDArray[np.floating],
  statistic: Callable[[NDArray[np.floating]], float],
  samples: int,
  block_size: int,
  seed: int = 0,
):
  np.random.seed(seed)

  statistic_store = np.zeros(samples)
  data_size = data.shape[0]
  num_blocks = int(np.ceil(float(data_size) / block_size))

  for s in range(samples):
    data_ = np.concatenate(
      [
        data[j : j + block_size]
        for j in np.random.randint(0, data_size - block_size, num_blocks)
      ]
    )[:data_size]
    statistic_store[s] = statistic(data_)

  statistic_mean = np.mean(statistic_store)

  return BootstrapResult(
    statistic=float(statistic_mean),
    bias=float(statistic_mean - statistic(data)),
    standard_error=float(np.std(statistic_store)),
  )


def blocking(data: NDArray[np.floating]):

  data_size = len(data)

  d = int(np.log2(data_size))
  s = np.zeros(d)
  gamma = np.zeros(d)
  mean = np.mean(data)

  for i in range(d):
    # Autocovariance at lag 1
    gamma[i] = np.mean((data[:-1] - mean) * (data[1:] - mean))

    # Variance
    s[i] = np.var(data)

    # Blocking transformation
    data = 0.5 * (data[0::2] + data[1::2])

  M = (np.cumsum(((gamma / s) ** 2 * 2 * np.arange(1, d + 1)[::-1])[::-1]))[::-1]

  # Chi-square critical values
  q = np.array(
    [
      6.634897,
      9.210340,
      11.344867,
      13.276704,
      15.086272,
      16.811894,
      18.475307,
      20.090235,
      21.665994,
      23.209251,
      24.724970,
      26.216967,
      27.688250,
      29.141238,
      30.577914,
      31.999927,
      33.408664,
      34.805306,
      36.190869,
      37.566235,
      38.932173,
      40.289360,
      41.638398,
      42.979820,
      44.314105,
      45.641683,
      46.962942,
      48.278236,
      49.587884,
      50.892181,
    ]
  )

  k = d - 1
  for i in range(d):
    if M[i] < q[i]:
      k = i
      break

  if k >= d - 1:
    print("Warning: Use more data")

  variance = s[k] / 2 ** (d - k)

  return mean, variance
