from typing import NamedTuple
from dataclasses import dataclass

from numba import njit
import numpy as np
from numpy.typing import NDArray
from structs import ParameterGrid, ParamConstructor


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

    x = positions[i, 0]
    y = positions[i, 1]
    z = positions[i, 2]

    r2_gauss = x * x + y * y + beta2 * z * z
    single_particle_term += -2.0 * alpha * (2.0 + beta) + 4.0 * alpha2 * r2_gauss

    elliptic_trap += x * x + y * y + gamma2 * z * z

    for j in range(n):
      if i == j:
        continue

      xj = positions[j, 0]
      yj = positions[j, 1]
      zj = positions[j, 2]

      dx = xi - xj
      dy = yi - yj
      dz = zi - zj

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

    # term_3 = sum_i ||sum_j u_ij||^2
    term_3 += u_vec[0] * u_vec[0] + u_vec[1] * u_vec[1] + u_vec[2] * u_vec[2]

  log_laplacian = (
    single_particle_term - 4.0 * alpha * a * term_2 + a2 * (term_3 - term_4)
  )
  energy = 0.5 * (-log_laplacian + elliptic_trap)

  return energy


@njit(fastmath=True)
def local_energy_(positions: NDArray[np.floating], params: BoseParams):
  alpha, beta, gamma, a = params
  n = positions.shape[0]

  alpha2 = alpha**2
  beta2 = beta**2
  gamma2 = gamma**2
  a2 = a**2

  single_particle_term = np.sum(
    -2.0 * alpha * (2.0 + beta)
    + 4.0
    * alpha2
    * (positions[:, 0] ** 2 + positions[:, 1] ** 2 + beta2 * positions[:, 2] ** 2)
  )

  term_2 = 0.0
  term_3 = 0.0
  term_4 = 0.0

  u_vec = np.empty(3, dtype=np.float64)
  for i in range(n):
    u_vec[0] = 0.0
    u_vec[1] = 0.0
    u_vec[2] = 0.0

    for j in range(n):
      if i == j:
        continue

      rij_vec = positions[i] - positions[j]
      rij = np.linalg.norm(rij_vec)
      if rij <= a:
        return np.inf

      rij2 = rij**2
      inv_dist = 1.0 / (rij2 * (rij - a))

      nominator_2 = (
        rij_vec[0] * positions[i, 0]
        + rij_vec[1] * positions[i, 1]
        + beta * rij_vec[2] * positions[i, 2]
      )
      term_2 += nominator_2 * inv_dist

      term_4 += 1.0 / (rij2 * (rij - a) ** 2)

      u_vec += rij_vec * inv_dist

    term_3 += np.dot(u_vec, u_vec)

  log_laplacian = (
    single_particle_term - 4.0 * alpha * a * term_2 + a2 * (term_3 - term_4)
  )

  elliptic_trap = np.sum(
    positions[:, 0] ** 2 + positions[:, 1] ** 2 + gamma2 * positions[:, 2] ** 2
  )

  energy = 0.5 * (-log_laplacian + elliptic_trap)

  return energy
