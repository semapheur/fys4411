from typing import Callable, NamedTuple

import jax.numpy as jnp
import numpy as np
import optax
from jax import Array, debug, jit, lax, random, tree, tree_util, vmap
from numpy.typing import NDArray
from structs import ParameterGrid
from vmc import GridSearchResult

type CycleCarryBrute = tuple[Array, Array, Array, Array]
type CycleCarryImportance = tuple[Array, Array, Array, Array, Array]
type CycleCarryOptimize[P: NamedTuple] = tuple[Array, Array, Array, Array, P, P]
type CarryGradientDescent[P: NamedTuple] = tuple[
  Array, P, P, optax.OptState, Array, Array, Array
]


class OptimizationResult[P: NamedTuple](NamedTuple):
  energy_history: NDArray[np.floating]
  variance_history: NDArray[np.floating]
  error_history: NDArray[np.floating]
  parameters: P

@jit(static_argnums=(0, 1, 2, 3, 5, 6))
def metropolis_step_jax[P: NamedTuple](
  dimensions: int,
  particles: int,
  log_wavefunction: Callable[[Array, P], Array],
  local_energy: Callable[[Array, P], Array],
  parameters: P,
  step_size: float,
  cycles: int,
  rng_key: jnp.ndarray,
) -> tuple[Array, Array, Array]:
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

  # Initialize variables
  rng_key, init_key = random.split(rng_key)

  position_0 = step_size * (
    random.uniform(init_key, (particles, dimensions)) - 0.5
  )
  wf_0 = log_wavefunction(position_0, parameters)

  def cycle_step(
    carry: CycleCarryBrute, carry_key: Array
  ) -> tuple[CycleCarryBrute, Array]:

    position_old, wf_old, energy, energy2 = carry

    # Split the random key
    carry_key, walk_key, accept_key = random.split(carry_key, 3)

    # Propose new configuration
    delta = step_size * (random.uniform(walk_key, position_old.shape) - 0.5)
    position_new = position_old + delta

    # Calculate probability amplitude for proposed configuration
    wf_new = log_wavefunction(position_new, parameters)

    # Calculate acceptance ratio for the move
    log_acceptance_ratio = 2 * (wf_new - wf_old)

    # Update position and probability amplitude for accepted moves
    accept = jnp.log(random.uniform(accept_key)) < log_acceptance_ratio
    position_old = jnp.where(accept, position_new, position_old)
    wf_old = jnp.where(accept, wf_new, wf_old)

    energy_delta = local_energy(position_old, parameters)
    energy_new = energy + energy_delta
    energy2_new = energy2 + energy_delta**2

    return (position_old, wf_old, energy_new, energy2_new), energy_new

  # Initialize carry
  zero_scalar = jnp.array(0.0)  # Initial energy
  carry_0 = (position_0, wf_0, zero_scalar, zero_scalar)

  # Iterate over all Monte Carlo cycles
  keys = random.split(rng_key, cycles)
  (_, _, energy, energy2), energy_accummulator = lax.scan(
    cycle_step, carry_0, keys
  )

  cycle_counts = jnp.arange(1, cycles + 1)
  energies = energy_accummulator / cycle_counts

  return energy / cycles, energy2 / cycles, energies

@jit(static_argnums=(0, 1, 2, 3, 4, 6, 7, 8))
def metropolis_step_importance_jax[P: NamedTuple](
  dimensions: int,
  particles: int,
  log_wavefunction: Callable[[Array, P], Array],
  local_energy: Callable[[Array, P], Array],
  drift_force: Callable[[Array, P], Array],
  parameters: P,
  time_step: float,
  diffusion_coefficient: float,
  cycles: int,
  rng_key: Array,
) -> tuple[Array, Array, Array]:

    # Precompute constants
  dt_sqrt = jnp.sqrt(time_step)
  dt_D = time_step * diffusion_coefficient

  rng_key, init_key, cycle_key = random.split(rng_key, 3)

  # Initialize positions with Gaussian noise
  position_0 = (
    random.normal(init_key, (particles, dimensions)) * dt_sqrt
  )
  wf_0 = log_wavefunction(position_0, parameters)
  force_0 = drift_force(position_0, parameters)

  def cycle_step(
    carry: CycleCarryImportance, carry_key: Array
  ) -> tuple[CycleCarryImportance, Array]:

    position_old, wf_old, force_old, energy, energy2 = carry

    walk_key, accept_key = random.split(carry_key)

    # Propose new configuration
    delta = random.normal(walk_key, (position_old.shape)) * dt_sqrt + force_old * dt_D
    position_new = position_old + delta

    # Calculate probability amplitude and quantum force for proposed configuration
    wf_new = log_wavefunction(position_new, parameters)
    force_new = drift_force(position_new, parameters)

    # Calculate Green's function exponent
    force_sum = force_old + force_new
    force_diff = force_old - force_new
    pos_diff = position_old - position_new
    greens_exponent = jnp.sum(0.5 * force_sum * (0.5 * dt_D * force_diff + pos_diff))

    # Calculate acceptance ratio for the move
    log_acceptance_ratio = greens_exponent + 2 * (wf_new - wf_old)

    # Update position and probability amplitude for accepted moves
    accept = jnp.log(random.uniform(accept_key)) < log_acceptance_ratio
    position_old = jnp.where(accept, position_new, position_old)
    wf_old = jnp.where(accept, wf_new, wf_old)
    force_old = jnp.where(accept, force_new, force_old)

    energy_delta = local_energy(position_old, parameters)
    energy_new = energy + energy_delta
    energy2_new = energy2 + energy_delta**2

    return (
      position_old,
      wf_old,
      force_old,
      energy_new,
      energy2_new,
    ), energy_new

  # Initalize carry for Monte Carlo cycles
  zero_scalar = jnp.array(0.0)  # Initial energy
  carry_0 = (
    position_0,
    wf_0,
    force_0,
    zero_scalar,
    zero_scalar,
  )

  # Iterate over all Monte Carlo cycles
  cycle_keys = random.split(cycle_key, cycles)
  (_, _, _, energy, energy2), energy_accumulator = lax.scan(
    cycle_step,
    carry_0,
    cycle_keys,
  )

  cycle_counts = jnp.arange(1, cycles + 1)
  energies = energy_accumulator / cycle_counts

  return energy / cycles, energy2 / cycles, energies

@jit(static_argnums=(0, 1, 2, 3, 4, 5, 7, 8, 9))
def metropolis_step_optimization_jax[P: NamedTuple](
  dimensions: int,
  particles: int,
  log_wavefunction: Callable[[Array, P], Array],
  wavefunction_derivative: Callable[[Array, P], P],
  local_energy: Callable[[Array, P], Array],
  drift_force: Callable[[Array, P], Array],
  parameters: P,
  time_step: float,
  diffusion_coefficient: float,
  cycles: int,
  rng_key: Array,
) -> tuple[Array, Array]:

  # Precompute constants
  dt_sqrt = jnp.sqrt(time_step)
  dt_D = time_step * diffusion_coefficient

  # Initialize random keys
  rng_key, init_key, cycle_key = random.split(rng_key, 3)

  # Initialize variables
  position_0 = (
    random.normal(init_key, (particles, dimensions)) * dt_sqrt
  )
  wf_0 = log_wavefunction(position_0, parameters)
  force_0 = drift_force(position_0, parameters)

  def cycle_step(carry: CycleCarryOptimize[P], carry_key: Array):
    position_old, wf_old, force_old, energy, psi_derivative, psi_e_derivative = carry

    walk_key, accept_key = random.split(carry_key)

    # Propose new configuration
    delta = random.normal(walk_key, position_old.shape) * dt_sqrt + force_old * dt_D
    position_new = position_old + delta

    # Calculate probability amplitude and quantum force
    wf_new = log_wavefunction(position_new, parameters)
    force_new = drift_force(position_new, parameters)

    # Calculate Green's function exponent
    force_sum = force_old + force_new
    force_diff = force_old - force_new
    pos_diff = position_old - position_new
    greens_exponent = jnp.sum(0.5 * force_sum * (0.5 * dt_D * force_diff * pos_diff))

    # Calculate acceptance ratio for the move
    log_acceptance_ratio = greens_exponent + 2 * (wf_new - wf_old)

    # Update position, probability amplitude and quantum force for accepted moves
    accept = jnp.log(random.uniform(accept_key)) < log_acceptance_ratio
    position_old = jnp.where(accept, position_new, position_old)
    wf_old = jnp.where(accept, wf_new, wf_old)
    force_old = jnp.where(accept, force_new, force_old)

    # Update energy and energy gradient terms
    energy_delta = local_energy(position_old, parameters)
    psi_delta = wavefunction_derivative(position_old, parameters)

    psi_derivative = tree.map(lambda a, b: a + b, psi_derivative, psi_delta)
    psi_e_derivative = tree.map(
      lambda a, b: a + b * energy_delta, psi_e_derivative, psi_delta
    )

    return (
      position_old,
      wf_old,
      force_old,
      energy + energy_delta,
      psi_derivative,
      psi_e_derivative,
    ), None

  # Initialize carry for Monte Carlo cycles
  zero_params = tree.map(lambda x: jnp.zeros_like(x), parameters)
  carry_0 = (position_0, wf_0, force_0, jnp.array(0.0), zero_params, zero_params)

  # Iterate over all Monte Carlo cycles
  cycle_keys = random.split(cycle_key, cycles)
  (final_state, _) = lax.scan(cycle_step, carry_0, cycle_keys)
  _, _, _, total_energy, total_psi_delta, total_psi_e = final_state

  # Calculate mean energy and energy gradient
  energy_mean = total_energy / cycles
  psi_derivative_mean = tree.map(lambda x: x / cycles, total_psi_delta)
  psi_e_mean = tree.map(lambda x: x / cycles, total_psi_e)

  energy_gradient = tree.map(
    lambda pe, pd: 2 * (pe - pd * energy_mean), psi_e_mean, psi_derivative_mean
  )

  return energy_mean, energy_gradient

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

  def sample_importance(
    self,
    wavefunction: Callable[[Array, P], Array],
    local_energy: Callable[[Array, P], Array],
    drift_force: Callable[[Array, P], Array],
    parameters: P,
    time_step: float,
    diffusion_coefficient: float,
    cycles: int,
    seed: int = 0,
  ) -> tuple[Array, Array, Array]:
    rng_key = random.PRNGKey(seed)

    return metropolis_step_importance_jax(
      self.dimensions,
      self.number_particles,
      wavefunction,
      local_energy,
      drift_force,
      parameters,
      time_step,
      diffusion_coefficient,
      cycles,
      rng_key,
    )

  def _grid_search[PG: ParameterGrid[P]](
    self,
    parameter_grid: PG,
    cycles: int,
    run_one: Callable[[P, Array], tuple[Array, Array, Array]],
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
    energies, energies2, _ = batched_run(batched_params, rng_keys)

    variances = energies2 - energies**2
    error = jnp.sqrt(jnp.abs(variances) / cycles)

    return GridSearchResult(
      params=params_list,
      energies=np.array(energies),
      variances=np.array(variances),
      errors=np.array(error),
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

    def run_one(params_slice: P, rng_key: Array) -> tuple[Array, Array, Array]:
      return metropolis_step_jax(
        self.dimensions,
        self.number_particles,
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

    def run_one(params_slice: P, rng_key: Array) -> tuple[Array, Array, Array]:
      return metropolis_step_importance_jax(
        self.dimensions,
        self.number_particles,
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

  @jit(static_argnums=(0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13))
  def optimize(
    self,
    log_wavefunction: Callable[[Array, P], Array],
    wavefunction_derivative: Callable[[Array, P], Array],
    local_energy: Callable[[Array, P], Array],
    drift_force: Callable[[Array, P], Array],
    parameters: P,
    time_step: float,
    diffusion_coefficient: float,
    cycles: int,
    optimization_iterations: int,
    optimizer: optax.GradientTransformation,
    gradient_tolerance: float = 1e-2,
    ema_decay: float = 0.95,
    seed: int = 0,
  ):
    optimizer_clipped = optax.chain(
      optax.clip_by_global_norm(1.0),
      optimizer
    )
    optimizer_state = optimizer_clipped.init(parameters)
    rng = random.PRNGKey(seed)

    def condition(carry: CarryGradientDescent[P]):
      i, _, _, _, _, _, gradient_norm = carry
      return (gradient_norm > gradient_tolerance) & (i < optimization_iterations)

    def update_step(carry: CarryGradientDescent[P]):
      i, params, params_ema, opt_state, rng, _, _ = carry

      key_step, rng_next = random.split(rng)

      energy, energy_gradient = metropolis_step_optimization_jax(
        self.dimensions,
        self.number_particles,
        log_wavefunction,
        wavefunction_derivative,
        local_energy,
        drift_force,
        params,
        time_step,
        diffusion_coefficient,
        cycles,
        key_step,
      )

      # Calculate gradient norm to test convergence
      gradient_norm = optax.global_norm(energy_gradient)

      # Update optimizer state and parameters
      updates, opt_state_next = optimizer_clipped.update(energy_gradient, opt_state, params)
      params_next = optax.apply_updates(params, updates)

      # Exponential moving average update to smooth out stochastic noise
      params_ema_next = tree_util.tree_map(
        lambda ema, p: ema_decay * ema + (1.0 - ema_decay)  * p, params_ema, params_next
      )


      # Print progress
      def print_fn():
        debug.print(
          "iteration {i}/{t}: E={e:.2f}, grad={g:.3e}, {p}",
          i=i,
          t=optimization_iterations,
          e=energy,
          g=gradient_norm,
          p=params_ema_next,
        )

      lax.cond(i % 10 == 0, print_fn, lambda: None)

      return i + 1, params_next, params_ema_next, opt_state_next, rng_next, energy, gradient_norm

    carry_init = (0, parameters, parameters, optimizer_state, rng, jnp.array(0.0), jnp.array(1.0))
    final_state = (
      lax.while_loop(
        condition,
        update_step,
        carry_init,
      )
    )
    final_iteration, params_final, params_ema_final, _, _, energy_final, gradient_norm_final = final_state

    debug.print(
      "Finished at iteration {i}: E (ema)={e:.2f}, grad={g:.3e}, {p}",
      i=final_iteration,
      e=energy_final,
      g=gradient_norm_final,
      p=params_final,
    )

    return energy_final, params_ema_final
