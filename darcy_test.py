# Author: Jonathan Siegel
#
# Tests the Darcy solver.

import numpy as np
from matplotlib import pyplot as plt
from examples.data_gen import uniform_grid_solver_scalar, sample_gaussian_field, sample_scalar_diffusion, sample_piecewise_constant_scalar_diffusion

if __name__ == '__main__':
  n, m = (256, 256)
  a = sample_scalar_diffusion(3*n, 3*m)
  plt.imshow(np.log(a), extent=[0,2*np.pi,0,2*np.pi])
  plt.colorbar()
  plt.show()
  for i in range(4):
    f = 2.0 * sample_gaussian_field(n, m, width=0.4)
    plt.imshow(f, extent=[0,2*np.pi,0,2*np.pi])
    plt.colorbar()
    plt.show()

    # Move to a 2pi x 2pi grid.
    u = (2.0 * np.pi * 2.0 * np.pi) * uniform_grid_solver_scalar(a, f)
    plt.imshow(u, extent=[0,2*np.pi,0,2*np.pi])
    plt.colorbar()
    plt.show()
