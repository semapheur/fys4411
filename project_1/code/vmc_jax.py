from typing import Callable, NamedTuple

import jax.numpy as jnp
import numpy as np
import optax
from jax import Array, jit, lax, random, tree_util, vmap
from numpy.typing import NDArray
from structs import ParameterGrid

type ParticleCarryBrute = tuple[Array, Array]
type CycleCarryBrute = tuple[Array, Array, Array, Array]
type ParticleCarryImportance = tuple[Array, Array, Array]
type CycleCarryImportance = tuple[Array, Array, Array, Array, Array]
type CycleCarryOptimize = tuple[Array, Array, Array, Array, Array, Array, Array]


class GridSearchResult[P: NamedTuple](NamedTuple):
  params: list[P]
  energy: NDArray[np.floating]
  variance: NDArray[np.floating]
  error: NDArray[np.floating]


class OptimizationResult[P: NamedTuple](NamedTuple):
  energy_history: NDArray[np.floating]
  variance_history: NDArray[np.floating]
  error_history: NDArray[np.floating]
  parameters: P


class MetropolisJAX[P: NamedTuple]:
  """
  Class for performing Monte Carlo simulations using the Metropolis algorithm. This class utilizes JAX for vectorization.

  Attributes
  ----------
  number_particles : int
    Number of particles in the system.
  dimensions : int
    Dimension of the system.
  param_type : type[P]
    Type of the parameters used to evaluate the wave function and local energy.
  """

  def __init__(self, number_particles: int, dimensions: int):
    self.number_particles = number_particles
    self.dimensions = dimensions

  @jit(static_argnums=(0, 1, 2, 4, 5))
  def _step(
    self,
    wavefunction: Callable[[Array, P], Array],
    local_energy: Callable[[Array, P], Array],
    parameters: P,
    step_size: float,
    cycles: int,
    rng_key: jnp.ndarray,
  ) -> tuple[Array, Array]:
    """
    Perform a single Metropolis Monte Carlo step. The function is JIT-compiled with static arguments using JAX

    Parameters
    ----------
    wavefunction : Callable[[Array, P, Array]
      Wave function used to calculate the probability amplitude.
    local_energy : Callable[[Array, P, Array]
      Local energy function used to calculate the energy.
    parameters : P
      Parameters used to evaluate the wave function and local energy.
    step_size : float
      Step size used to update the position.
    cycles : int
      Number of Monte Carlo cycles to perform.
    rng_key : jnp.ndarray
      Random key used to generate random numbers.

    Returns
    -------
    energy : float
      Calculated energy.
    energy_squared : float
      Calculated energy squared.
    """
    position_initial = jnp.zeros((self.number_particles, self.dimensions))
    wf_intial = wavefunction(position_initial, parameters)

    def cycle_step(
      carry: CycleCarryBrute, carry_key: Array
    ) -> tuple[CycleCarryBrute, None]:
      """
      JAX-vectorized helper function to perform a Monte Carlo cycle.
      """
      pos_old, wf_old, energy, energy_squared = carry

      def per_particle(
        state: ParticleCarryBrute, i: Array
      ) -> tuple[ParticleCarryBrute, None]:
        """
        JAX-vectorized helper function to perform a Metropolis Monte Carlo step for a single particle.
        """

        pos_old, wf_old = state

        # Split the random key
        particle_key = random.fold_in(carry_key, i)
        walk_key, accept_key = random.split(particle_key)

        # Walk position
        delta = step_size * (random.uniform(walk_key, (self.dimensions,)) - 0.5)
        pos_new = pos_old.at[i].add(delta)

        # Calculate acceptance ratio for the move
        wf_new = wavefunction(pos_new, parameters)
        acceptance_ratio = (wf_new / wf_old) ** 2

        # Update position and probability amplitude for accepted moves
        accept = random.uniform(accept_key) < acceptance_ratio
        pos_old = jnp.where(accept, pos_new, pos_old)
        wf_old = jnp.where(accept, wf_new, wf_old)

        return (pos_old, wf_old), None

      # Iterate over all particles
      (pos_old, wf_old), _ = lax.scan(
        per_particle, (pos_old, wf_old), jnp.arange(self.number_particles)
      )

      energy_delta = local_energy(pos_old, parameters)
      return (
        pos_old,
        wf_old,
        energy + energy_delta,
        energy_squared + energy_delta**2,
      ), None

    cycle_keys = random.split(rng_key, cycles)
    zero = jnp.zeros(())  # Initial energy

    # Iterate over all Monte Carlo cycles
    (_, _, energy, energy_squared), _ = lax.scan(
      cycle_step, (position_initial, wf_intial, zero, zero), cycle_keys
    )

    return energy / cycles, energy_squared / cycles

  @jit(static_argnums=(0, 1, 2, 3, 5, 6, 7))
  def _step_importance(
    self,
    wavefunction: Callable[[Array, P], Array],
    local_energy: Callable[[Array, P], Array],
    drift_force: Callable[[Array, P], Array],
    parameters: P,
    time_step: float,
    diffusion_coefficient: float,
    cycles: int,
    rng_key: Array,
  ):

    # Precompute constants
    dt_sqrt = jnp.sqrt(time_step)
    dt_D = time_step * diffusion_coefficient

    rng_key, init_key, cycle_key = random.split(rng_key, 3)

    # Initialize positions with Gaussian noise
    position_initial = (
      random.normal(init_key, (self.number_particles, self.dimensions)) * dt_sqrt
    )
    wf_intial = wavefunction(position_initial, parameters)
    force_initial = drift_force(position_initial, parameters)

    def cycle_step(
      carry: CycleCarryImportance, carry_key: Array
    ) -> tuple[CycleCarryImportance, None]:
      """
      JAX-vectorized helper function to perform a Monte Carlo cycle.
      """
      pos_old, wf_old, force_old, energy, energy_squared = carry

      def per_particle(
        state: ParticleCarryImportance, i: Array
      ) -> tuple[ParticleCarryImportance, None]:
        """
        JAX-vectorized helper function to perform a Metropolis Monte Carlo step for a single particle.
        """

        pos_old, wf_old, force_old = state

        # Split the random key
        particle_key = random.fold_in(carry_key, i)
        walk_key, accept_key = random.split(particle_key)

        # Walk position
        delta = (
          random.normal(walk_key, (self.dimensions,)) * dt_sqrt + force_old[i] * dt_D
        )
        pos_new = pos_old.at[i].add(delta)

        wf_new = wavefunction(pos_new, parameters)
        force_new = drift_force(pos_new, parameters)

        # Calculate Green's function exponent
        force_sum = force_old[i] + force_new[i]
        force_diff = force_old[i] - force_new[i]
        pos_delta = pos_new[i] - pos_old[i]
        greens_exponent = jnp.sum(
          0.5 * force_sum * (0.5 * dt_D * force_diff - pos_delta)
        )

        # Calculate acceptance ratio for the move
        acceptance_ratio = jnp.exp(greens_exponent) * (wf_new / wf_old) ** 2

        # Update position and probability amplitude for accepted moves
        accept = random.uniform(accept_key) < acceptance_ratio
        pos_old = jnp.where(accept, pos_new, pos_old)
        wf_old = jnp.where(accept, wf_new, wf_old)
        force_old = jnp.where(accept, force_new, force_old)

        return (pos_old, wf_old, force_old), None

      # Iterate over all particles
      (
        (
          pos_old,
          wf_old,
          force_old,
        ),
        _,
      ) = lax.scan(
        per_particle, (pos_old, wf_old, force_old), jnp.arange(self.number_particles)
      )

      energy_delta = local_energy(pos_old, parameters)
      return (
        pos_old,
        wf_old,
        force_old,
        energy + energy_delta,
        energy_squared + energy_delta**2,
      ), None

    cycle_keys = random.split(cycle_key, cycles)
    zero = jnp.zeros(())  # Initial energy

    # Iterate over all Monte Carlo cycles
    (_, _, _, energy, energy_squared), _ = lax.scan(
      cycle_step, (position_initial, wf_intial, force_initial, zero, zero), cycle_keys
    )

    return energy / cycles, energy_squared / cycles

  @jit(static_argnums=(0, 1, 2, 3, 4, 6, 7, 8))
  def _step_optimization(
    self,
    wavefunction: Callable[[Array, P], Array],
    wavefunction_derivative: Callable[[Array, P], Array],
    local_energy: Callable[[Array, P], Array],
    drift_force: Callable[[Array, P], Array],
    parameters: P,
    time_step: float,
    diffusion_coefficient: float,
    cycles: int,
    rng_key: Array,
  ):
    # Precompute constants
    dt_sqrt = jnp.sqrt(time_step)
    dt_D = time_step * diffusion_coefficient

    rng_key, init_key, cycle_key = random.split(rng_key, 3)

    # Initialize positions with Gaussian noise
    position_initial = (
      random.normal(init_key, (self.number_particles, self.dimensions)) * dt_sqrt
    )
    wf_intial = wavefunction(position_initial, parameters)
    force_initial = drift_force(position_initial, parameters)

    def cycle_step(
      carry: CycleCarryOptimize, carry_key: Array
    ) -> tuple[CycleCarryOptimize, None]:
      """
      JAX-vectorized helper function to perform a Monte Carlo cycle.
      """
      (
        pos_old,
        wf_old,
        force_old,
        energy,
        energy_squared,
        psi_delta,
        psi_e_derivative,
      ) = carry

      def per_particle(
        state: ParticleCarryImportance, i: Array
      ) -> tuple[ParticleCarryImportance, None]:
        """
        JAX-vectorized helper function to perform a Metropolis Monte Carlo step for a single particle.
        """

        pos_old, wf_old, force_old = state

        # Split the random key
        particle_key = random.fold_in(carry_key, i)
        walk_key, accept_key = random.split(particle_key)

        # Walk position
        delta = (
          random.normal(walk_key, (self.dimensions,)) * dt_sqrt + force_old[i] * dt_D
        )
        pos_new = pos_old.at[i].add(delta)

        wf_new = wavefunction(pos_new, parameters)
        force_new = drift_force(pos_new, parameters)

        # Calculate Green's function exponent
        force_sum = force_old[i] + force_new[i]
        force_diff = force_old[i] - force_new[i]
        pos_delta = pos_new[i] - pos_old[i]
        greens_exponent = jnp.sum(
          0.5 * force_sum * (0.5 * dt_D * force_diff - pos_delta)
        )

        # Calculate acceptance ratio for the move
        acceptance_ratio = jnp.exp(greens_exponent) * (wf_new / wf_old) ** 2

        # Update position and probability amplitude for accepted moves
        accept = random.uniform(accept_key) < acceptance_ratio
        pos_old = jnp.where(accept, pos_new, pos_old)
        wf_old = jnp.where(accept, wf_new, wf_old)
        force_old = jnp.where(accept, force_new, force_old)

        return (pos_old, wf_old, force_old), None

      # Iterate over all particles
      (
        (
          pos_old,
          wf_old,
          force_old,
        ),
        _,
      ) = lax.scan(
        per_particle, (pos_old, wf_old, force_old), jnp.arange(self.number_particles)
      )

      energy_delta = local_energy(pos_old, parameters)
      psi_derivative = wavefunction_derivative(pos_old, parameters)

      return (
        pos_old,
        wf_old,
        force_old,
        energy + energy_delta,
        energy_squared + energy_delta**2,
        psi_delta + psi_derivative,
        psi_e_derivative + psi_derivative * energy_delta,
      ), None

    cycle_keys = random.split(cycle_key, cycles)
    zero_scalar = jnp.zeros(())  # Initial energy
    zero_array = jnp.zeros((len(parameters),))

    # Iterate over all Monte Carlo cycles
    (_, _, _, energy, energy_squared, psi_delta, psi_e_derivative), _ = lax.scan(
      cycle_step,
      (
        position_initial,
        wf_intial,
        force_initial,
        zero_scalar,
        zero_scalar,
        zero_array,
        zero_array,
      ),
      cycle_keys,
    )

    energy /= cycles
    energy_squared /= cycles
    psi_delta /= cycles
    psi_e_derivative /= cycles
    energy_derivative = 2 * (psi_e_derivative - psi_delta * energy)

    return energy, energy_squared, energy_derivative

  def _grid_search[PG: ParameterGrid[P]](
    self,
    parameter_grid: PG,
    cycles: int,
    run_one: Callable[[P, Array], tuple[Array, Array]],
    seed: int = 0,
  ) -> GridSearchResult[P]:
    params_list = parameter_grid.combos()

    # Convert to JAX arrays
    batched_params = tree_util.tree_map(lambda *xs: jnp.array(xs), *params_list)

    # Generate random keys
    rng = random.PRNGKey(seed)
    rng_keys = random.split(rng, len(params_list))

    # Vectorize Monte Carlo simulation
    batched_run = vmap(run_one)

    # Run Monte Carlo simulation
    energies, energies_squared = batched_run(batched_params, rng_keys)
    variances = energies_squared - energies**2
    error = jnp.sqrt(jnp.abs(variances) / cycles)

    return GridSearchResult(
      params=params_list,
      energy=np.array(energies),
      variance=np.array(variances),
      error=np.array(error),
    )

  def grid_search_brute[PG: ParameterGrid[P]](
    self,
    wavefunction: Callable[[Array, P], Array],
    local_energy: Callable[[Array, P], Array],
    parameter_grid: PG,
    step_size: float,
    cycles: int,
    seed: int = 0,
  ) -> GridSearchResult[P]:
    """
    Perform a grid search over the parameter space.

    Parameters
    ----------
    wavefunction : Callable[[Array, P], Array]
      Wave function used to calculate the probability amplitude.
    local_energy : Callable[[Array, P], Array]
      Local energy function used to calculate the energy.
    parameter_grid : dict[str, np.ndarray]
      Parameter grid used for the grid search.
    step_size : float
      Step size used to update the position.
    cycles : int
      Number of Monte Carlo cycles to perform.
    seed : int
      Seed used to generate random numbers.

    Returns
    -------
    GridSearchResult
      Result containing the parameters, energy, variance, and error.
    """

    def run_one(params_slice: P, rng_key: Array) -> tuple[Array, Array]:
      return self._step(
        wavefunction,
        local_energy,
        params_slice,
        step_size,
        cycles,
        rng_key,
      )

    return self._grid_search(parameter_grid, cycles, run_one, seed)

  def grid_search_importance[PG: ParameterGrid[P]](
    self,
    wavefunction: Callable[[Array, P], Array],
    local_energy: Callable[[Array, P], Array],
    drift_force: Callable[[Array, P], Array],
    parameter_grid: PG,
    time_step: float,
    diffusion_coefficient: float,
    cycles: int,
    seed: int = 0,
  ) -> GridSearchResult[P]:
    """
    Perform a grid search over the parameter space using importance sampling.

    Parameters
    ----------
    wavefunction : Callable[[Array, P], Array]
      Wave function used to calculate the probability amplitude.
    local_energy : Callable[[Array, P], Array]
      Local energy function used to calculate the energy.
    parameter_grid : PG
      Parameter grid used for the grid search.
    time_step : float
      Time step used to update the position.
    diffusion_coefficient : float
      Diffusion coefficient used to update the position.
    cycles : int
      Number of Monte Carlo cycles to perform.
    seed : int
      Seed used to generate random numbers.

    Returns
    -------
    GridSearchResult
      Result containing the parameters, energy, variance, and error.
    """

    def run_one(params_slice: P, rng_key: Array) -> tuple[Array, Array]:
      return self._step_importance(
        wavefunction,
        local_energy,
        drift_force,
        params_slice,
        time_step,
        diffusion_coefficient,
        cycles,
        rng_key,
      )

    return self._grid_search(parameter_grid, cycles, run_one, seed)

  def optimize(
    self,
    wavefunction: Callable[[Array, P], Array],
    wavefunction_derivative: Callable[[Array, P], Array],
    local_energy: Callable[[Array, P], Array],
    drift_force: Callable[[Array, P], Array],
    parameters: P,
    time_step: float,
    diffusion_coefficient: float,
    cycles: int,
    optimization_iterations: int,
    optimizer: optax.GradientTransformation,
    seed: int = 0,
  ):

    ParamType = type(parameters)
    param_fields = parameters._fields
    param_array = jnp.array([getattr(parameters, field) for field in param_fields])

    optimizer_state = optimizer.init(param_array)
    rng = random.PRNGKey(seed)

    energy_history: list[float] = []
    variance_history: list[float] = []
    error_history: list[float] = []

    best_energy = float("inf")
    best_params = parameters

    for i in range(optimization_iterations):
      rng, key = random.split(rng)

      energy, energy_squared, grad = self._step_optimization(
        wavefunction,
        wavefunction_derivative,
        local_energy,
        drift_force,
        parameters,
        time_step,
        diffusion_coefficient,
        cycles,
        key,
      )

      updates, optimizer_state = optimizer.update(grad, optimizer_state, param_array)
      param_array = optax.apply_updates(param_array, updates)
      parameters = ParamType(*param_array)

      energy_ = float(energy)
      energy_variance = float(energy_squared - energy_**2)
      energy_error = np.sqrt(np.abs(energy_variance) / cycles)

      energy_history.append(energy_)
      variance_history.append(energy_variance)
      error_history.append(energy_error)

      if energy_ < best_energy:
        best_energy = energy_
        best_params = parameters

    return OptimizationResult(
      energy_history=np.array(energy_history),
      variance_history=np.array(variance_history),
      error_history=np.array(error_history),
      parameters=best_params,
    )
