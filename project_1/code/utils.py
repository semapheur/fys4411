from typing import Callable, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array, vmap
from numba import njit
from numpy.typing import NDArray


@njit(fastmath=True)
def gradient_finite_diff[P: NamedTuple](
  function: Callable[[NDArray[np.floating], P], float],
  x: NDArray[np.floating],
  params: P,
  eps: float = 1e-3,
):
  x_flat = x.flatten()
  n = x_flat.shape[0]

  gradient = np.zeros(n, dtype=np.float64)
  inv_2eps = 1.0 / (2 * eps)

  for i in range(n):
    x_work = x_flat.copy()
    x_old = x_work[i]

    x_work[i] = x_old + eps
    f_plus = function(x_work.reshape(x.shape), params)

    x_work[i] = x_old - eps
    f_minus = function(x_work.reshape(x.shape), params)

    gradient[i] = (f_plus - f_minus) * inv_2eps

  return gradient.reshape(x.shape)


@njit(fastmath=True)
def laplacian_finite_diff[P: NamedTuple](
  function: Callable[[NDArray[np.floating], P], float],
  x: NDArray[np.floating],
  params: P,
  eps: float = 1e-3,
):
  x_flat = x.flatten()

  n = x_flat.shape[0]
  f0 = function(x, params)

  laplacian = 0.0

  inv_eps2 = 1.0 / eps**2

  for i in range(n):
    x_old = x_flat[i]

    x_flat[i] = x_old + eps
    f_plus = function(x_flat.reshape(x.shape), params)

    x_flat[i] = x_old - eps
    f_minus = function(x_flat.reshape(x.shape), params)

    x_flat[i] = x_old
    laplacian += (f_plus - 2 * f0 + f_minus) * inv_eps2

  return laplacian


def gradient_finite_diff_jax(
  function: Callable[[Array], Array],
  x: Array,
  eps: float = 1e-3,
) -> Array:
  x_flat = x.flatten()
  inv_2eps = 1.0 / (2 * eps)

  gradient = jnp.zeros_like(x_flat)

  def first_partial(i: int) -> Array:
    e_i = jnp.zeros_like(x_flat).at[i].set(1.0)  # Unit vector in the i-th direction
    f_plus = function((x_flat + eps * e_i).reshape(x.shape))
    f_minus = function((x_flat - eps * e_i).reshape(x.shape))
    return (f_plus - f_minus) * inv_2eps

  indices = jnp.arange(x_flat.shape[0])
  gradient = vmap(first_partial)(indices)

  return gradient.reshape(x.shape)


def laplacian_finite_diff_jax(
  function: Callable[[Array], Array],
  x: Array,
  eps: float = 1e-3,
) -> Array:
  """
  Compute the Laplacian of a given function using finite differences.

  Parameters
  ----------
  function : Callable[[Array], Array]
    Scalar function vectorized with JAX
  x : Array
    The vector input at which to compute the Laplacian
  eps : float, optional
    The finite difference step size (default 1e-4)

  Returns
  -------
  The Laplacian of `function` at `x`
  """
  x_flat = x.flatten()
  f0 = function(x)
  inv_eps2 = 1.0 / eps**2

  def second_partial(i: int) -> Array:
    e_i = jnp.zeros_like(x_flat).at[i].set(1.0)  # Unit vector in the i-th direction
    f_plus = function((x_flat + eps * e_i).reshape(x.shape))
    f_minus = function((x_flat - eps * e_i).reshape(x.shape))
    return (f_plus - 2 * f0 + f_minus) * inv_eps2

  indices = jnp.arange(x_flat.shape[0])
  partials = vmap(second_partial)(indices)

  return jnp.sum(partials)
