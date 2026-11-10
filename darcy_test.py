# Author: Jonathan Siegel
#
# Tests the Darcy solver.

import numpy as np
from matplotlib import pyplot as plt
from examples.data_gen import uniform_grid_solver_scalar

if __name__ == '__main__':
  n, m = (256, 256)
  f = np.ones((n, m))
  # xf = np.linspace(1.0 / (n+2), 1.0 - 1.0 / (n+2), num=n)
  # yf = np.linspace(1.0 / (m+2), 1.0 - 1.0 / (m+2), num=m)
  # xa = np.linspace(0.0, 1.0, num=3*n)
  # ya = np.linspace(0.0, 1.0, num=3*m)
  a = np.ones((3*n, 3*m))
  a[n:3*n, m:2*m] = 100
  plt.imshow(f)
  plt.show()
  plt.imshow(a)
  plt.show()
  u = uniform_grid_solver_scalar(a, f)
  plt.imshow(u)
  plt.show()
