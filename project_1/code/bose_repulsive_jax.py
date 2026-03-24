from typing import NamedTuple

from jax import Array, debug
import jax.numpy as jnp


class BoseParams(NamedTuple):
  alpha: Array
  beta: Array
  gamma: Array
  a: Array


def wavefunction_jax(positions: Array, params: BoseParams) -> Array:
  alpha, beta, _, a = params
  num_particles = positions.shape[0]

  x2 = positions[:, 0] ** 2
  y2 = positions[:, 1] ** 2
  z2 = positions[:, 2] ** 2

  # Single particle factor
  g = jnp.exp(-alpha * (x2 + y2 + beta * z2))
  single_particle = jnp.prod(g)

  # Pairwise distances
  rij_vec = positions[:, None, :] - positions[None, :, :]
  rij = jnp.linalg.norm(rij_vec, axis=-1)

  i_idx, j_idx = jnp.triu_indices(num_particles, k=1)
  rij_pairs = rij[i_idx, j_idx]

  # Hard-core condition
  violates = jnp.any(rij_pairs <= a)

  # Jastrow factor
  jastrow_factor = jnp.prod(1.0 - a / rij_pairs)

  # Wavefunction
  psi = single_particle * jastrow_factor
  psi = jnp.where(violates, 0.0, psi)  # chceck for violations

  return psi


def wavefunction_derivative_jax(positions: Array, params: BoseParams) -> BoseParams:
  _, beta, _, _ = params

  alpha_derivative = -jnp.sum(
    positions[:, 0] ** 2 + positions[:, 1] ** 2 + beta * positions[:, 2] ** 2
  )

  return BoseParams(
    alpha=alpha_derivative, beta=jnp.array(0.0), gamma=jnp.array(0.0), a=jnp.array(0.0)
  )


def local_energy_jax(positions: Array, params: BoseParams) -> Array:
  alpha, beta, gamma, a = params
  num_particles = positions.shape[0]

  # Precompute parameter squares
  alpha2 = alpha**2
  beta2 = beta**2
  gamma2 = gamma**2
  a2 = a**2

  # Single particle term
  r2_xy = positions[:, 0] ** 2 + positions[:, 1] ** 2
  r2_z = positions[:, 2] ** 2

  single_particle_term = jnp.sum(
    -2.0 * alpha * (2.0 + beta) + 4.0 * alpha2 * (r2_xy + beta2 * r2_z)
  )

  # Pairwise distances
  rij_vec = positions[:, None, :] - positions[None, :, :]
  rij = jnp.linalg.norm(rij_vec, axis=-1)
  rij2 = rij**2

  # Hard-core condition
  off_diagonal = ~jnp.eye(num_particles, dtype=bool)  # mask out diagonal
  violates = jnp.any((rij <= a) & off_diagonal)

  safe_dist = jnp.where(off_diagonal, rij, 1.0)

  # Cross terms
  inv_dist = 1.0 / (safe_dist**2 * (safe_dist - a))
  nominator_2 = (
    rij_vec[..., 0] * positions[:, 0, None]
    + rij_vec[..., 1] * positions[:, 1, None]
    + beta * rij_vec[..., 2] * positions[:, 2, None]
  )
  term_2 = jnp.sum(jnp.where(off_diagonal, nominator_2 * inv_dist, 0.0))

  term_4 = jnp.sum(jnp.where(off_diagonal, 1.0 / (rij2 * (rij - a) ** 2), 0.0))

  u_vec = jnp.sum(
    jnp.where(off_diagonal[:, :, None], rij_vec * inv_dist[..., None], 0.0), axis=1
  )
  term_3 = jnp.sum(jnp.sum(u_vec**2, axis=-1))

  # Logarithmic Laplacian
  log_laplacian = (
    single_particle_term - 4.0 * alpha * a * term_2 + a2 * (term_3 - term_4)
  )

  # Elliptic trap potential
  elliptic_trap = jnp.sum(
    positions[:, 0] ** 2 + positions[:, 1] ** 2 + gamma2 * positions[:, 2] ** 2
  )

  # Local energy
  energy = 0.5 * (-log_laplacian + elliptic_trap)
  energy = jnp.where(violates, jnp.inf, energy)  # check for hard-core violation

  return energy


def drift_force_jax(positions: Array, params: BoseParams):
  alpha, beta, _, a = params
  num_particles = positions.shape[0]

  scale = jnp.array([1.0, 1.0, beta])
  external_force = -4.0 * alpha * positions * scale

  rij_vec = positions[:, None, :] - positions[None, :, :]
  rij2 = jnp.sum(rij_vec**2, axis=-1)
  rij = jnp.linalg.norm(rij_vec)

  off_diagonal = ~jnp.eye(num_particles, dtype=bool)
  violates = jnp.any((rij <= a) & off_diagonal)
  inv = jnp.where(off_diagonal, 1.0 / (rij2 * (rij - a) ** 2), 0.0)
  repulsive_force = 2 * a * jnp.sum(rij * inv[..., None], axis=1)

  forces = external_force + repulsive_force
  return jnp.where(violates, jnp.zeros_like(forces), forces)
