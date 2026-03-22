from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
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

  wf = wavefunction_jax(positions, params)

  def wf_partial(pos: Array) -> Array:
    return wavefunction_jax(pos, params)

  wf_laplacian = laplacian_finite_diff_jax(wf_partial, positions)

  potential_external = 0.5 * jnp.sum(positions**2)
  kinetic = -0.5 * wf_laplacian / wf

  return kinetic + potential_external


def drift_force_jax(position: Array, params: HarmonicParams) -> Array:
  alpha = params.alpha
  return -4 * alpha * position


def drift_force_numeric_jax(position: Array, params: HarmonicParams) -> Array:

  wf = wavefunction_jax(position, params)

  def wf_partial(pos: Array) -> Array:
    return wavefunction_jax(pos, params)

  wf_gradient = gradient_finite_diff_jax(wf_partial, position)

  return 2 * wf_gradient / wf
