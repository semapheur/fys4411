from typing import NamedTuple

import jax.numpy as jnp
from jax import Array, grad, jvp, vmap
from utils import gradient_finite_diff_jax, laplacian_finite_diff_jax


class HarmonicParams(NamedTuple):
  alpha: Array


def wavefunction_jax(positions: Array, params: HarmonicParams) -> Array:
  alpha = params.alpha
  r2 = jnp.sum(positions**2)

  return jnp.exp(-alpha * r2)

def log_wavefunction_jax(positions: Array, params: HarmonicParams) -> Array:
  alpha = params.alpha
  r2 = jnp.sum(positions**2)

  return -alpha * r2


def wavefunction_derivative_jax(
  positions: Array, params: HarmonicParams
) -> HarmonicParams:
  r2 = jnp.sum(positions**2)

  return HarmonicParams(alpha=-r2)


def local_energy_jax(positions: Array, params: HarmonicParams) -> Array:
  alpha = params.alpha
  number_particles = positions.shape[0]
  dimensions = positions.shape[1]

  r2 = jnp.sum(positions**2)
  e = dimensions * number_particles * alpha + (0.5 - 2 * alpha**2) * r2
  return e


def local_energy_numeric_jax(positions: Array, params: HarmonicParams) -> Array:
  """
  Computes the local energy numerically using this formula for logarithmic Laplacian (∇^2 Ψ/Ψ) = ∇^2 (ln Ψ) + |∇ ln Ψ|^2
  """

  def log_wf_partial(pos: Array) -> Array:
    return log_wavefunction_jax(pos, params)
  
  wf_log_grad = grad(log_wf_partial)(positions)

  def log_wf_grad_flat(pos: Array) -> Array:
    return grad(log_wf_partial)(pos.reshape(positions.shape)).reshape(-1)

  positions_flat = positions.reshape(-1) 
  dim = positions_flat.size
  eye = jnp.eye(dim)
  _, hessian_rows = vmap(lambda t: jvp(log_wf_grad_flat, (positions_flat,), (t,)))(eye)

  wf_log_laplacian = jnp.trace(hessian_rows)

  potential_external = jnp.sum(positions**2)
  kinetic = wf_log_laplacian  + jnp.sum(wf_log_grad**2)

  return 0.5 * (-kinetic + potential_external)


def drift_force_jax(positions: Array, params: HarmonicParams) -> Array:
  alpha = params.alpha
  return -4 * alpha * positions


def drift_force_numeric_jax(positions: Array, params: HarmonicParams) -> Array:
  """
  Computes the quantum force using the formula F = 2 * ∇ ln Ψ 
  """

  def log_wf_partial(pos: Array) -> Array:
    return log_wavefunction_jax(pos, params)

  return 2 * grad(log_wf_partial)(positions)
