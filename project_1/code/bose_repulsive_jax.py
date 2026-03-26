from typing import NamedTuple

from jax import Array, ops, vmap
import jax.numpy as jnp


class BoseParams(NamedTuple):
  alpha: Array
  beta: Array
  gamma: Array
  a: Array


def wavefunction_jax(positions: Array, params: BoseParams):
  alpha, beta, _, a = params

  # Single particle factor
  r2_weighted = (
    positions[:, 0] ** 2 + positions[:, 1] ** 2 + beta * positions[:, 2] ** 2
  )
  log_single_particle = -alpha * jnp.sum(r2_weighted)

  # Pairwise Jastrow factor using vmap
  def jastrow_factor(position_i: Array, all_positions: Array):
    # Distance calculation
    rij_vec = position_i - all_positions
    rij = jnp.sqrt(jnp.sum(rij_vec**2, axis=-1))

    # Mask self-interaction (where rij == 0)
    mask = rij > 0

    # Stability: Ensure we don't take log of negative or div by zero
    safe_rij = jnp.where(mask, rij, a + 1.0)

    # log(1 - a/r)
    jastrow_terms = jnp.where(mask, jnp.log1p(-a / safe_rij), 0.0)

    # Check violations (only for other particles)
    violation_found = jnp.any((rij <= a) & mask)

    return jnp.sum(jastrow_terms), violation_found

  # Map the scan over all particles
  log_jastrow_vec, violates_vec = vmap(jastrow_factor, in_axes=(0, None))(
    positions, positions
  )

  # Since we summed all i,j (where i != j), we have to divide by 2
  log_jastrow_factor = 0.5 * jnp.sum(log_jastrow_vec)
  violates = jnp.any(violates_vec)

  # Compute wavefunction
  log_psi = log_single_particle + log_jastrow_factor
  psi = jnp.exp(log_psi)

  return jnp.where(violates, 0.0, psi)


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

  # Particle interaction terms
  def particle_interactions(position_i: Array, all_positions: Array):
    rij_vec = position_i - all_positions
    rij2 = jnp.sum(rij_vec**2, axis=-1)
    rij = jnp.sqrt(rij2)

    # Mask self-interaction (where rij == 0)
    mask = rij > 0

    # Ensure stability by avoiding negative log or division by zero
    safe_rij = jnp.where(mask, rij, a + 1.0)

    # Compute cross terms
    inv_factor = jnp.where(mask, 1.0 / (rij2 * (safe_rij - a)), 0.0)

    nominator_2 = (
      rij_vec[:, 0] * position_i[0]
      + rij_vec[:, 1] * position_i[1]
      + beta * rij_vec[:, 2] * position_i[2]
    )
    term_2_i = jnp.sum(nominator_2 * inv_factor)

    uij2 = rij2 * inv_factor**2
    term_3_i = jnp.sum(uij2)

    term_4_i = jnp.sum(jnp.where(mask, 1.0 / (rij2 * (safe_rij - a) ** 2), 0.0))

    violates_i = jnp.any((rij <= a) & mask)

    return term_2_i, term_3_i, term_4_i, violates_i

  term_2_vec, term_3_vec, term_4_vec, violates_vec = vmap(
    particle_interactions, in_axes=(0, None)
  )(positions, positions)

  term_2 = jnp.sum(term_2_vec)
  term_3 = jnp.sum(term_3_vec)
  term_4 = jnp.sum(term_4_vec)
  violates = jnp.any(violates_vec)

  # Logarithmic Laplacian
  log_laplacian = (
    single_particle_term - 4.0 * alpha * a * term_2 + a2 * (term_3 - term_4)
  )

  # Elliptic trap potential
  elliptic_trap = jnp.sum(r2_xy + gamma2 * r2_z)

  # Local energy
  energy = 0.5 * (-log_laplacian + elliptic_trap)

  return jnp.where(violates, jnp.inf, energy)  # check for hard-core violation


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
