import numpy as np
from numpy.typing import NDArray


def radial_density(
  positions: NDArray[np.floating], max_radius: float, bins: int, bin_scaling: float
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:

  flat_positions = positions.reshape(-1, positions.shape[-1])
  radii = np.linalg.norm(flat_positions, axis=1)

  x = np.linspace(0.0, 1.0, bins + 1)
  bin_edges = max_radius * x**bin_scaling

  counts, _ = np.histogram(radii, bins=bin_edges)

  bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

  # shell volume for each spherical shell
  shell_volume = (4 / 3) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)

  n_samples = flat_positions.shape[0]
  density = counts / (n_samples * shell_volume)

  return bin_centers, density
