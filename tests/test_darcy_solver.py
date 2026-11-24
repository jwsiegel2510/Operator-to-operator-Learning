# Tests for the Darcy solver.

import unittest
import numpy as np
from examples.data_gen import uniform_grid_solver_scalar

def interior_grid(n):
  x = np.arange(1, n + 1) / (n + 1)
  return np.meshgrid(x, x, indexing='ij')

def nodal_grid(n):
  x = np.linspace(0.0, 1.0, n)
  return np.meshgrid(x, x, indexing='ij')

class TestDarcySolver(unittest.TestCase):
  def test_constant_coefficient_is_exact_on_eigenfunction(self):
    """a = 1 with a sine eigenfunction is solved exactly by the sine Galerkin method."""
    n = 15
    X, Y = interior_grid(n)
    exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    solution = uniform_grid_solver_scalar(np.ones((n + 2, n + 2)), 2 * np.pi**2 * exact, tol=1e-12)
    self.assertLess(np.abs(solution - exact).max(), 1e-10)

  def test_solution_scales_with_forcing(self):
    """The tolerance is relative, so scaling f scales the solution and nothing else."""
    n = 15
    X, Y = interior_grid(n)
    XN, YN = nodal_grid(n + 2)
    a = 2.0 + 0.5 * np.sin(2 * np.pi * XN) * np.sin(2 * np.pi * YN)
    f = np.sin(np.pi * X) * np.sin(3 * np.pi * Y) + 1.0
    base = uniform_grid_solver_scalar(a, f)
    for scale in (1e-3, 1e3):
      scaled = uniform_grid_solver_scalar(a, scale * f)
      self.assertLess(np.abs(scaled - scale * base).max() / np.abs(scale * base).max(), 1e-9)

  def test_high_contrast_coefficient_agrees_with_tight_tolerance(self):
    """A 10^4 contrast disc converges to the same answer as a much tighter tolerance."""
    n = 23
    XN, YN = nodal_grid(n + 2)
    radius = np.sqrt((XN - 0.5)**2 + (YN - 0.5)**2)
    a = np.where(radius < 0.3, 1e4, 1.0)
    f = np.ones((n, n))
    default = uniform_grid_solver_scalar(a, f)
    tight = uniform_grid_solver_scalar(a, f, tol=1e-11)
    self.assertLess(np.abs(default - tight).max() / np.abs(tight).max(), 1e-5)
    # f >= 0 and a > 0, so the maximum principle gives u >= 0 up to roundoff.
    self.assertGreater(tight.min(), -1e-10 * np.abs(tight).max())

  def test_rejects_invalid_coefficient(self):
    n = 15
    a = np.ones((n + 2, n + 2))
    a[3, 4] = 0.0
    with self.assertRaises(ValueError):
      uniform_grid_solver_scalar(a, np.ones((n, n)))

  def test_raises_when_iteration_limit_is_reached(self):
    n = 31
    XN, YN = nodal_grid(n + 2)
    a = np.where(XN > 0.5, np.exp(6.0), np.exp(-6.0))
    with self.assertRaises(RuntimeError):
      uniform_grid_solver_scalar(a, np.ones((n, n)), maxiter=2)

if __name__ == '__main__':
  unittest.main()
