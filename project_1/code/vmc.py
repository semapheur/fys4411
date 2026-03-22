from typing import Callable, NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray
from stats import seed_numba
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


@njit(fastmath=True)
def metropolis_step_numba[P: NamedTuple](
  wavefunction: ScalarFunction[P],
  local_energy: ScalarFunction[P],
  parameters: P,
  step_size: float,
  cycles: int,
  number_particles: int,
  dimension: int,
) -> tuple[float, NDArray[np.floating]]:
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
  """

  # Initialize variables
  positions = np.zeros((number_particles, dimension), dtype=np.float64)
  row_store = np.empty(dimension, dtype=np.float64)

  wf_old = wavefunction(positions, parameters)

  energy = 0.0
  energies = np.empty(cycles, dtype=np.float64)

  # Perform Monte Carlo cycles
  for cycle in range(cycles):
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
    energies[cycle] = energy / (cycle + 1)

  energy /= cycles

  return energy, energies


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
) -> tuple[float, NDArray[np.floating]]:
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
  """

  # Precompute constants
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
  energies = np.empty(cycles, dtype=np.float64)

  # Perform Monte Carlo cycles
  for cycle in range(cycles):
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
      force_new[:] = drift_force(positions, parameters)

      # Calculate Green's function
      greens_exponent = 0.0
      for j in range(dimension):
        force_sum = force_old[i, j] + force_new[i, j]
        force_diff = force_old[i, j] - force_new[i, j]
        position_delta = positions[i, j] - row_store[j]
        greens_exponent += 0.5 * force_sum * (0.5 * dt_D * force_diff - position_delta)

      # Calculate acceptance ratio for the move
      acceptance_ratio = np.exp(greens_exponent) * (wf_new / wf_old) ** 2

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
    energies[cycle] = energy / (cycle + 1)

  energy /= cycles

  return energy, energies


@njit(fastmath=True)
def metropolis_step_optimization_numba[P: NamedTuple](
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

  # Precompute constants
  dt_sqrt = np.sqrt(time_step)
  dt_D = time_step * diffusion_coefficient

  # Initialize positions with Gaussian noise
  positions = np.zeros((number_particles, dimension), dtype=np.float64)
  for i in range(number_particles):
    for j in range(dimension):
      positions[i, j] = np.random.randn() * dt_sqrt

  # Initialize variables
  position_old = np.empty(dimension, dtype=np.float64)
  force_old = drift_force(positions, parameters)
  force_new = np.empty_like(force_old)
  wf_old = wavefunction(positions, parameters)

  energy = 0.0
  psi_delta = np.zeros(len(parameters), dtype=np.float64)
  psi_e_derivative = np.zeros(len(parameters), dtype=np.float64)

  # Perform Monte Carlo cycles
  for _ in range(cycles):
    for i in range(number_particles):
      # Store current position of particle i before proposing move
      for j in range(dimension):
        position_old[j] = positions[i, j]

      # Propose move for particle i using Langevin dynamics
      for j in range(dimension):
        positions[i, j] = (
          position_old[j] + np.random.randn() * dt_sqrt + force_old[i, j] * dt_D
        )

      wf_new = wavefunction(positions, parameters)
      force_new[:] = drift_force(positions, parameters)

      # Calculate Green's function
      greens_exponent = 0.0
      for j in range(dimension):
        force_sum = force_old[i, j] + force_new[i, j]
        force_diff = force_old[i, j] - force_new[i, j]
        position_diff = position_old[j] - positions[i, j]
        greens_exponent += 0.5 * force_sum * (0.5 * dt_D * force_diff + position_diff)

      # Calculate acceptance ratio for the move
      acceptance_ratio = np.exp(greens_exponent) * (wf_new / wf_old) ** 2

      if np.random.rand() < acceptance_ratio:  # Accept move
        wf_old = wf_new
        for j in range(dimension):
          force_old[i, j] = force_new[i, j]

      else:  # Reject move
        for j in range(dimension):
          positions[i, j] = position_old[j]

    # Calculate energy
    energy_delta = local_energy(positions, parameters)
    psi_derivative = wavefunction_derivative(positions, parameters)
    energy += energy_delta
    psi_delta += psi_derivative
    psi_e_derivative += psi_derivative * energy_delta

  cycles_inv = 1.0 / cycles
  energy *= cycles_inv
  psi_delta *= cycles_inv
  psi_e_derivative *= cycles_inv
  energy_gradient = 2 * (psi_e_derivative - psi_delta * energy)

  return energy, energy_gradient


class Metropolis[P: NamedTuple]:
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

  def __init__(self, number_particles: int, dimension: int):
    self.number_particles = number_particles
    self.dimension = dimension

  def sample_importance(
    self,
    wavefunction: ScalarFunction[P],
    local_energy: ScalarFunction[P],
    drift_force: VectorFunction[P],
    parameters: P,
    time_step: float,
    diffusion_coefficient: float,
    cycles: int,
  ) -> tuple[float, NDArray[np.floating]]:
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

  def _grid_search[PG: ParameterGrid[P]](
    self,
    parameter_grid: PG,
    cycles: int,
    metropolis_step: Callable[[P], tuple[float, NDArray[np.floating]]],
  ) -> GridSearchResult[P]:
    params_list = parameter_grid.combos()

    energies_, energy_stores_ = zip(*(metropolis_step(p) for p in params_list))
    energies = np.array(energies_)
    energy_stores = np.vstack(energy_stores_)
    energies_squared = np.mean(energy_stores**2, axis=0)
    variances = energies_squared - energies**2
    error = np.sqrt(np.abs(variances) / cycles)

    return GridSearchResult(
      params=params_list,
      energy=energies,
      variance=variances,
      error=error,
    )

  def grid_search_brute[PG: ParameterGrid[P]](
    self,
    wavefunction: ScalarFunction[P],
    local_energy: ScalarFunction[P],
    parameter_grid: PG,
    step_size: float,
    cycles: int,
    seed: int = 0,
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

    seed_numba(seed)

    def metropolis_step(parameters: P) -> tuple[float, NDArray[np.floating]]:
      return metropolis_step_numba(
        wavefunction,
        local_energy,
        parameters,
        step_size,
        cycles,
        self.number_particles,
        self.dimension,
      )

    return self._grid_search(parameter_grid, cycles, metropolis_step)

  def grid_search_importance[PG: ParameterGrid[P]](
    self,
    wavefunction: ScalarFunction[P],
    local_energy: ScalarFunction[P],
    drift_force: VectorFunction[P],
    parameter_grid: PG,
    time_step: float,
    diffusion_coefficient: float,
    cycles: int,
    seed: int = 0,
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

    seed_numba(seed)

    def metropolis_step(parameters: P) -> tuple[float, NDArray[np.floating]]:
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

    return self._grid_search(parameter_grid, cycles, metropolis_step)

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
    optimization_iterations: int,
    learning_rate: float,
    gradient_clip: float = 1.0,
    gradient_tolerance: float = 1e-5,
    seed: int = 0,
  ):
    seed_numba(seed)

    # Extract parameter values
    param_fields = parameters._fields
    param_values = np.array(
      [getattr(parameters, field) for field in param_fields], dtype=np.float64
    )

    # Cache NamedTuple type
    ParamType = type(parameters)

    for i in range(optimization_iterations):
      energy, energy_gradient = metropolis_step_optimization_numba(
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

      gradient_norm = np.linalg.norm(energy_gradient)

      if gradient_norm < gradient_tolerance:
        break

      if gradient_norm > gradient_clip:
        energy_gradient *= gradient_clip / gradient_norm

      # Gradient descent steps
      param_values -= learning_rate * energy_gradient

      parameters = ParamType(*param_values)

      if (i + 1) % 10 == 0:
        print(
          f"iteration {i + 1}/{optimization_iterations}: E={energy:.2f}, grad={gradient_norm:.3e}, {parameters}"
        )

    print(f"Finished at {i} iterations")
    print(f"Energy={energy:.2f}, grad={gradient_norm:.3e}, {parameters}")

    return energy, parameters
