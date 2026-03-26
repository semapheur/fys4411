from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray
from structs import ParamConstructor, ParameterGrid


class BoseParams(NamedTuple):
  alpha: np.floating
  beta: np.floating
  gamma: np.floating
  a: np.floating


@dataclass(frozen=True)
class BoseParamGrid(ParameterGrid[BoseParams]):
  param_type: ParamConstructor[BoseParams]
  alpha: NDArray[np.floating]
  beta: NDArray[np.floating]
  gamma: NDArray[np.floating]
  a: NDArray[np.floating]


@njit(fastmath=True)
def wavefunction(positions: NDArray[np.floating], params: BoseParams):
  alpha, beta, _, a = params
  N = positions.shape[0]

  x2 = positions[:, 0] ** 2
  y2 = positions[:, 1] ** 2
  z2 = positions[:, 2] ** 2

  g = np.exp(-alpha * (x2 + y2 + beta * z2))

  jastrow_factor = 1.0

  for i in range(N):
    for j in range(i + 1, N):
      rij = np.linalg.norm(positions[i] - positions[j])
      if rij <= a:
        return 0.0

      jastrow_factor *= 1.0 - a / rij

  return jastrow_factor * np.prod(g)


@njit(fastmath=True)
def wavefunction_derivative(positions: NDArray[np.floating], params: BoseParams):
  _, beta, _, _ = params

  alpha_derivative = -np.sum(
    positions[:, 0] ** 2 + positions[:, 1] ** 2 + beta * positions[:, 2] ** 2
  )
  beta_derivative = 0
  gamma_derivative = 0
  a_derivative = 0

  return alpha_derivative, beta_derivative, gamma_derivative, a_derivative


@njit(fastmath=True)
def local_energy(positions: NDArray[np.floating], params: BoseParams):
  alpha, beta, gamma, a = params
  n = positions.shape[0]

  alpha2 = alpha * alpha
  beta2 = beta * beta
  gamma2 = gamma * gamma
  a2 = a * a

  single_particle_term = 0.0
  elliptic_trap = 0.0

  term_2 = 0.0
  term_3 = 0.0
  term_4 = 0.0

  u_vec = np.empty(3, dtype=np.float64)

  for i in range(n):
    xi = positions[i, 0]
    yi = positions[i, 1]
    zi = positions[i, 2]

    u_vec[0] = 0.0
    u_vec[1] = 0.0
    u_vec[2] = 0.0

    r2 = xi * xi + yi * yi + beta2 * zi * zi
    single_particle_term += -2.0 * alpha * (2.0 + beta) + 4.0 * alpha2 * r2

    elliptic_trap += xi * xi + yi * yi + gamma2 * zi * zi

    for j in range(n):
      if i == j:
        continue

      dx = xi - positions[j, 0]
      dy = yi - positions[j, 1]
      dz = zi - positions[j, 2]

      rij2 = dx * dx + dy * dy + dz * dz
      rij = np.sqrt(rij2)

      if rij <= a:
        return np.inf

      inv_rij2 = 1.0 / rij2
      inv = 1.0 / (rij2 * (rij - a))

      # term_2
      nominator_2 = dx * xi + dy * yi + beta * dz * zi
      term_2 += nominator_2 * inv

      # term_4
      diff = rij - a
      term_4 += inv_rij2 / (diff * diff)

      # accumulate u_vec
      u_vec[0] += dx * inv
      u_vec[1] += dy * inv
      u_vec[2] += dz * inv

    # term_3 = sum_i |sum_j u_ij|^2
    term_3 += u_vec[0] * u_vec[0] + u_vec[1] * u_vec[1] + u_vec[2] * u_vec[2]

  log_laplacian = (
    single_particle_term - 4.0 * alpha * a * term_2 + a2 * (term_3 - term_4)
  )
  energy = 0.5 * (-log_laplacian + elliptic_trap)

  return energy


@njit(fastmath=True)
def drift_force(positions: NDArray[np.floating], params: BoseParams):
  alpha, beta, _, a = params
  num_particles = positions.shape[0]

  forces = np.zeros((num_particles, 3))

  for i in range(num_particles):
    xi = positions[i, 0]
    yi = positions[i, 1]
    zi = positions[i, 2]

    # External force
    forces[i, 0] = -4 * alpha * xi
    forces[i, 1] = -4 * alpha * yi
    forces[i, 2] = -4 * alpha * beta * zi

    # Repulsive force
    for j in range(num_particles):
      if i == j:
        continue

      dx = xi - positions[j, 0]
      dy = yi - positions[j, 1]
      dz = zi - positions[j, 2]

      rij2 = dx * dx + dy * dy + dz * dz
      rij = np.sqrt(rij2)

      # Hard-core condition
      if rij <= a:
        return np.zeros((num_particles, 3))

      inv = 1.0 / (rij2 * (rij - a))

      forces[i, 0] += 2 * a * dx * inv
      forces[i, 1] += 2 * a * dy * inv
      forces[i, 2] += 2 * a * dz * inv

  return forces
