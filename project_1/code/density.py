import numpy as np
from numpy.typing import NDArray


def radial_onebody_density(
  positions: NDArray[np.floating], r_max: float, bins: int
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:

  flat_positions = positions.reshape(-1, positions.shape[-1])
  radial_distance = np.linalg.norm(flat_positions, axis=1)
  counts, bin_edges = np.histogram(radial_distance, bins=bins, range=(0.0, r_max))

  r_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
  dr = bin_edges[1] - bin_edges[0]

  # shell volume for each spherical shell
  shell_volume = 4 * np.pi * r_centers**2 * dr
  shell_volume[0] = np.inf  # avoid division by zero at r=0

  density = counts / shell_volume

  # normalize so ∫ρ(r) 4πr² dr = 1
  norm = np.sum(density[1:] * shell_volume[1:])
  density /= norm

  return r_centers, density
