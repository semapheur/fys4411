from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray
from structs import ParamConstructor, ParameterGrid
from utils import gradient_finite_diff, laplacian_finite_diff


class HarmonicParams(NamedTuple):
  alpha: np.floating


@dataclass(frozen=True)
class HarmonicParamGrid(ParameterGrid[HarmonicParams]):
  param_type: ParamConstructor[HarmonicParams]
  alpha: NDArray[np.floating]


@njit(fastmath=True)
def wavefunction(positions: NDArray[np.floating], params: HarmonicParams) -> float:
  alpha = params.alpha
  r2 = np.sum(positions**2)

  return np.exp(-alpha * r2)


@njit(fastmath=True)
def log_wavefunction(positions: NDArray[np.floating], params: HarmonicParams) -> float:
  alpha = params.alpha
  r2 = np.sum(positions**2)

  return -alpha * r2


@njit(fastmath=True)
def wavefunction_derivative(
  positions: NDArray[np.floating], params: HarmonicParams
) -> float:
  r2 = np.sum(positions**2)

  return -r2


@njit(fastmath=True)
def local_energy(positions: NDArray[np.floating], params: HarmonicParams) -> float:
  alpha = params.alpha
  number_particles: int = positions.shape[0]
  dimensions: int = positions.shape[1]

  r2 = np.sum(positions**2)
  e = dimensions * number_particles * alpha + (0.5 - 2 * alpha * alpha) * r2
  return float(e)


@njit(fastmath=True)
def local_energy_numeric(
  positions: NDArray[np.floating], params: HarmonicParams
) -> float:

  wf = wavefunction(positions, params)
  wf_laplacian = laplacian_finite_diff(wavefunction, positions, params)

  potential_external = 0.5 * np.sum(positions**2)
  kinetic = -0.5 * wf_laplacian / wf

  return float(kinetic + potential_external)


@njit(fastmath=True)
def drift_force(
  position: NDArray[np.floating], params: HarmonicParams
) -> NDArray[np.floating]:
  alpha = params.alpha
  return -4 * alpha * position


@njit(fastmath=True)
def drift_force_numeric(
  position: NDArray[np.floating], params: HarmonicParams
) -> NDArray[np.floating]:

  wf = wavefunction(position, params)
  wf_gradient = gradient_finite_diff(wavefunction, position, params)

  return 2 * wf_gradient / wf
