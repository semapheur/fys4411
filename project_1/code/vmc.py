from collections import namedtuple
from typing import Callable, NamedTuple

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from structs import ParameterGrid

type ScalarFunction[P: NamedTuple] = Callable[[NDArray[np.floating], P], float]
type VectorFunction[P: NamedTuple] = Callable[
  [NDArray[np.floating], P], NDArray[np.floating]
]


class GridSearchResult[P: NamedTuple](NamedTuple):
  params: list[P]
  energy: NDArray[np.floating]
  variance: NDArray[np.floating]
  error: NDArray[np.floating]


@njit
def seed_numba(s: int):
  """Numba helper function to seed the random number generator."""
  np.random.seed(s)


@njit(fastmath=True)
def metropolis_step_numba[P: NamedTuple](
  wavefunction: ScalarFunction[P],
  local_energy: ScalarFunction[P],
  parameters: P,
  step_size: float,
  cycles: int,
  number_particles: int,
  dimension: int,
) -> tuple[float, float]:
  """
  Perform a single Metropolis Monte Carlo step. JIT-compiled using numba

  Parameters
  ----------
  wavefunction : Callable[[NDArray[np.floating], P], float]
    Variational vave function used to calculate the probability amplitude.
  local_energy : Callable[[NDArray[np.floating], P], float]
    Local energy function used to calculate the energy.
  parameters : P
    Parameters used to evaluate the wave function and local energy.
  step_size : float
    Step size used to update the position.
  cycles : int
    Number of Monte Carlo cycles to perform.
  number_particles : int
    Number of particles in the system.
  dimension : int
    Dimension of the system.

  Returns
  -------
  energy : float
    Calculated energy.
  energy_squared : float
    Calculated energy squared.
  """

  # Initialize variables
  positions = np.zeros((number_particles, dimension), dtype=np.float64)
  row_store = np.empty(dimension, dtype=np.float64)

  energy = 0.0
  energy_squared = 0.0

  wf_old = wavefunction(positions, parameters)

  # Perform Monte Carlo cycles
  for _ in prange(cycles):
    for i in range(number_particles):
      # Store current position of particle i before proposing move
      for j in range(dimension):
        row_store[j] = positions[i, j]

      # Propose move for particle i
      for j in range(dimension):
        positions[i, j] = row_store[j] + step_size * (np.random.rand() - 0.5)

      # Calculate acceptance ratio for the move
      wf_new = wavefunction(positions, parameters)
      acceptance_ratio = (wf_new / wf_old) ** 2

      if np.random.rand() < acceptance_ratio:  # Accept move
        wf_old = wf_new
      else:  # Reject move
        for j in range(dimension):
          positions[i, j] = row_store[j]

    # Calculate energy
    energy_delta = local_energy(positions, parameters)
    energy += energy_delta
    energy_squared += energy_delta**2

  energy /= cycles
  energy_squared /= cycles

  return energy, energy_squared


@njit(fastmath=True)
def metropolis_step_importance_numba[P: NamedTuple](
  wavefunction: ScalarFunction[P],
  local_energy: ScalarFunction[P],
  drift_force: VectorFunction[P],
  parameters: P,
  time_step: float,
  diffusion_coefficient: float,
  cycles: int,
  number_particles: int,
  dimension: int,
):
  """
  Perform a single Metropolis Monte Carlo step using Langevin dynamics. JIT-compiled using numba

  Parameters
  ----------
  wavefunction : Callable[[NDArray[np.floating], P], float]
    Variational wave function used to calculate the probability amplitude.
  local_energy : Callable[[NDArray[np.floating], P], float]
    Local energy function used to calculate the energy.
  drift_force : Callable[[NDArray[np.floating], P], NDArray[np.floating]]
    Drift force function used to calculate the quantum force.
  parameters : P
    Parameters used to evaluate the wave function and local energy.
  time_step : float
    Time step used in the Langevin dynamics.
  diffusion_coefficient : float
    Diffusion coefficient used in the Langevin dynamics.
  cycles : int
    Number of Monte Carlo cycles to perform.
  number_particles : int
    Number of particles in the system.
  dimension : int
    Dimension of the system.

  Returns
  -------
  energy : float
    Calculated energy.
  energy_squared : float
    Calculated energy squared.
  """

  # Preccompyte constants
  dt_sqrt = np.sqrt(time_step)
  dt_D = time_step * diffusion_coefficient

  # Initialize positions with Gaussian noise
  positions = np.zeros((number_particles, dimension), dtype=np.float64)
  for i in range(number_particles):
    for j in range(dimension):
      positions[i, j] = np.random.randn() * dt_sqrt

  # Initialize variables
  row_store = np.empty(dimension, dtype=np.float64)
  force_old = drift_force(positions, parameters)
  force_new = np.empty_like(force_old)
  wf_old = wavefunction(positions, parameters)

  energy = 0.0
  energy_squared = 0.0

  # Perform Monte Carlo cycles
  for _ in prange(cycles):
    for i in range(number_particles):
      # Store current position of particle i before proposing move
      for j in range(dimension):
        row_store[j] = positions[i, j]

      # Propose move for particle i using Langevin dynamics
      for j in range(dimension):
        positions[i, j] = (
          row_store[j] + np.random.randn() * dt_sqrt + force_old[i, j] * dt_D
        )

      wf_new = wavefunction(positions, parameters)
      force_new = drift_force(positions, parameters)

      # Calculate Green's function
      greens_function = 0.0
      for j in range(dimension):
        force_sum = force_old[i, j] + force_new[i, j]
        displacement = positions[i, j] - row_store[j]
        greens_function += 0.5 * force_sum * (0.5 * dt_D * force_sum - displacement)

      # Calculate acceptance ratio for the move
      acceptance_ratio = np.exp(greens_function) * (wf_new / wf_old) ** 2

      if np.random.rand() < acceptance_ratio:  # Accept move
        wf_old = wf_new
        for d in range(dimension):
          force_old[i, d] = force_new[i, d]

      else:  # Reject move
        for d in range(dimension):
          positions[i, d] = row_store[d]

    # Calculate energy
    energy_delta = local_energy(positions, parameters)
    energy += energy_delta
    energy_squared += energy_delta**2

  energy /= cycles
  energy_squared /= cycles

  return energy, energy_squared


@njit(fastmath=True)
def metropolis_step_minimization_numba[P: NamedTuple](
  wavefunction: ScalarFunction[P],
  wavefunction_derivative: ScalarFunction[P],
  local_energy: ScalarFunction[P],
  drift_force: VectorFunction[P],
  parameters: P,
  time_step: float,
  diffusion_coefficient: float,
  cycles: int,
  number_particles: int,
  dimension: int,
):
  """
  Perform a single Metropolis Monte Carlo step using Langevin dynamics. JIT-compiled using numba

  Parameters
  ----------
  wavefunction : Callable[[NDArray[np.floating], P], float]
    Variational wave function used to calculate the probability amplitude.
  wavefunction_derivative : Callable[[NDArray[np.floating], P], float]
    Wave function used to calculate the probability amplitude.
  local_energy : Callable[[NDArray[np.floating], P], float]
    Local energy function used to calculate the energy.
  drift_force : Callable[[NDArray[np.floating], P], NDArray[np.floating]]
    Drift force function used to calculate the quantum force.
  parameters : P
    Parameters used to evaluate the wave function and local energy.
  time_step : float
    Time step used in the Langevin dynamics.
  diffusion_coefficient : float
    Diffusion coefficient used in the Langevin dynamics.
  cycles : int
    Number of Monte Carlo cycles to perform.
  number_particles : int
    Number of particles in the system.
  dimension : int
    Dimension of the system.

  Returns
  -------
  energy : float
    Calculated energy.
  energy_squared : float
    Calculated energy squared.
  """

  # Preccompyte constants
  dt_sqrt = np.sqrt(time_step)
  dt_D = time_step * diffusion_coefficient

  # Initialize positions with Gaussian noise
  positions = np.zeros((number_particles, dimension), dtype=np.float64)
  for i in range(number_particles):
    for j in range(dimension):
      positions[i, j] = np.random.randn() * dt_sqrt

  # Initialize variables
  row_store = np.empty(dimension, dtype=np.float64)
  force_old = drift_force(positions, parameters)
  force_new = np.empty_like(force_old)
  wf_old = wavefunction(positions, parameters)

  energy = 0.0
  psi_delta = np.zeros(len(parameters), dtype=np.float64)
  psi_e_derivative = np.zeros(len(parameters), dtype=np.float64)

  # Perform Monte Carlo cycles
  for _ in prange(cycles):
    for i in range(number_particles):
      # Store current position of particle i before proposing move
      for j in range(dimension):
        row_store[j] = positions[i, j]

      # Propose move for particle i using Langevin dynamics
      for j in range(dimension):
        positions[i, j] = (
          row_store[j] + np.random.randn() * dt_sqrt + force_old[i, j] * dt_D
        )

      wf_new = wavefunction(positions, parameters)
      force_new = drift_force(positions, parameters)

      # Calculate Green's function
      greens_function = 0.0
      for j in range(dimension):
        force_sum = force_old[i, j] + force_new[i, j]
        displacement = positions[i, j] - row_store[j]
        greens_function += 0.5 * force_sum * (0.5 * dt_D * force_sum - displacement)

      # Calculate acceptance ratio for the move
      acceptance_ratio = np.exp(greens_function) * (wf_new / wf_old) ** 2

      if np.random.rand() < acceptance_ratio:  # Accept move
        wf_old = wf_new
        for d in range(dimension):
          force_old[i, d] = force_new[i, d]

      else:  # Reject move
        for d in range(dimension):
          positions[i, d] = row_store[d]

    # Calculate energy
    energy_delta = local_energy(positions, parameters)
    psi_derivative = wavefunction_derivative(positions, parameters)
    energy += energy_delta
    psi_delta += psi_derivative
    psi_e_derivative += psi_derivative * energy_delta

  energy /= cycles
  psi_delta /= cycles
  psi_e_derivative /= cycles
  energy_derivative = 2 * (psi_e_derivative - psi_delta * energy)

  return energy, energy_derivative


class Metropolis[P: NamedTuple, PG: ParameterGrid]:
  """
  Class for performing Monte Carlo simulations using the Metropolis algorithm.

  Attributes
  ----------
  number_particles : int
    Number of particles in the system.
  dimensions : int
    Dimension of the system.
  seed : int
    Seed used to generate random numbers.
  """

  def __init__(self, number_particles: int, dimension: int, seed: int | None = None):
    self.number_particles = number_particles
    self.dimension = dimension

    if seed is not None:
      seed_numba(seed)

  def _step(
    self,
    wavefunction: ScalarFunction[P],
    local_energy: ScalarFunction[P],
    parameters: P,
    step_size: float,
    cycles: int,
  ) -> tuple[float, float]:
    """
    Perform a Monte Carlo simulation using the Metropolis algorithm.

    Parameters
    ----------
    wavefunction : Callable[[NDArray[np.floating], P], float]
      Wave function used to calculate the probability amplitude.
    local_energy : Callable[[NDArray[np.floating], P], float]
      Local energy function used to calculate the energy.
    parameters : P
      Parameters used to evaluate the wave function and local energy.
    step_size : float
      Step size used to update the position.
    cycles : int
      Number of Monte Carlo cycles to perform.

    Returns
    -------
    energy : float
      Calculated energy.
    energy_squared : float
      Calculated energy squared.
    """
    return metropolis_step_numba(
      wavefunction,
      local_energy,
      parameters,
      step_size,
      cycles,
      self.number_particles,
      self.dimension,
    )

  def _step_importance(
    self,
    wavefunction: ScalarFunction[P],
    local_energy: ScalarFunction[P],
    drift_force: VectorFunction[P],
    parameters: P,
    time_step: float,
    diffusion_coefficient: float,
    cycles: int,
  ):
    """
    Perform a Monte Carlo simulation using the Metropolis algorithm with importance sampling.

    Parameters
    ----------
    wavefunction : Callable[[NDArray[np.floating], P], float]
      Wave function used to calculate the probability amplitude.
    local_energy : Callable[[NDArray[np.floating], P], float]
      Local energy function used to calculate the energy.
    drift_force : Callable[[NDArray[np.floating], P], NDArray[np.floating]]
      Drift force function used to calculate the drift force.
    parameters : P
      Parameters used to evaluate the wave function, local energy, and drift force.
    time_step : float
      Time step used to update the position.
    diffusion_coefficient : float
      Diffusion coefficient used to update the position.
    cycles : int
      Number of Monte Carlo cycles to perform.

    Returns
    -------
    energy : float
      Calculated energy.
    energy_squared : float
      Calculated energy squared.
    """

    return metropolis_step_importance_numba(
      wavefunction,
      local_energy,
      drift_force,
      parameters,
      time_step,
      diffusion_coefficient,
      cycles,
      self.number_particles,
      self.dimension,
    )

  def _grid_search(
    self,
    parameter_grid: PG,
    cycles: int,
    run_one: Callable[[P], tuple[float, float]],
  ) -> GridSearchResult[P]:
    params_list = parameter_grid.combos()

    energies_, energies_squared_ = zip(*(run_one(p) for p in params_list))
    energies = np.array(energies_)
    energies_squared = np.array(energies_squared_)
    variances = energies_squared - energies**2
    error = np.sqrt(np.abs(variances) / cycles)

    return GridSearchResult(
      params=params_list,
      energy=energies,
      variance=variances,
      error=error,
    )

  def grid_search_brute(
    self,
    wavefunction: ScalarFunction[P],
    local_energy: ScalarFunction[P],
    parameter_grid: PG,
    step_size: float,
    cycles: int,
  ) -> GridSearchResult[P]:
    """
    Perform a grid search over the parameter space.

    Parameters
    ----------
    wavefunction : Callable[[NDArray[np.floating], P], float]
      Wave function used to calculate the probability amplitude.
    local_energy : Callable[[NDArray[np.floating], P], float]
      Local energy function used to calculate the energy.
    parameter_grid : PG
      Parameter grid used for the grid search.
    step_size : float
      Step size used to update the position.
    cycles : int
      Number of Monte Carlo cycles to perform.

    Returns
    -------
    results : list[GridSearchResult[P]]
      List of results containing the parameters, energy, variance, and error.
    """

    def run_one(parameters: P) -> tuple[float, float]:
      return self._step(wavefunction, local_energy, parameters, step_size, cycles)

    return self._grid_search(parameter_grid, cycles, run_one)

  def grid_search_importance(
    self,
    wavefunction: ScalarFunction[P],
    local_energy: ScalarFunction[P],
    drift_force: VectorFunction[P],
    parameter_grid: PG,
    time_step: float,
    diffusion_coefficient: float,
    cycles: int,
  ) -> GridSearchResult[P]:
    """
    Perform a grid search over the parameter space.

    Parameters
    ----------
    wavefunction : Callable[[NDArray[np.floating], P], float]
      Wave function used to calculate the probability amplitude.
    local_energy : Callable[[NDArray[np.floating], P], float]
      Local energy function used to calculate the energy.
    drift_force : Callable[[NDArray[np.floating], P], NDArray[np.floating]]
      Drift force function used to calculate the drift force.
    parameter_grid : PG
      Parameter grid used for the grid search.
    time_step : float
      Time step used to update the position.
    diffusion_coefficient : float
      Diffusion coefficient used to update the position.
    cycles : int
      Number of Monte Carlo cycles to perform.

    Returns
    -------
    results : list[GridSearchResult[P]]
      List of results containing the parameters, energy, variance, and error.
    """

    def run_one(parameters: P) -> tuple[float, float]:
      return self._step_importance(
        wavefunction,
        local_energy,
        drift_force,
        parameters,
        time_step,
        diffusion_coefficient,
        cycles,
      )

    return self._grid_search(parameter_grid, cycles, run_one)

  def optimize(
    self,
    wavefunction: ScalarFunction[P],
    wavefunction_derivative: ScalarFunction[P],
    local_energy: ScalarFunction[P],
    drift_force: VectorFunction[P],
    parameters: P,
    time_step: float,
    diffusion_coefficient: float,
    cycles: int,
    learning_rate: float,
    optimization_iterations: int,
  ):

    parameters_ = np.array(parameters._asdict().values())

    for _ in range(optimization_iterations):
      energy, energy_derivative = metropolis_step_minimization_numba(
        wavefunction,
        wavefunction_derivative,
        local_energy,
        drift_force,
        parameters,
        time_step,
        diffusion_coefficient,
        cycles,
        self.number_particles,
        self.dimension,
      )

      parameters_ -= learning_rate * energy_derivative
      parameters = namedtuple(**zip(parameters._fields, parameters))

    return energy, parameters
