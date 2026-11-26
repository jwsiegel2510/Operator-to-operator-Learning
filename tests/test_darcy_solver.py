# Tests for the Darcy solver.

import unittest
import numpy as np
from examples.data_gen import uniform_grid_solver_scalar, uniform_grid_solver_tensor

def interior_grid(n):
  x = np.arange(1, n + 1) / (n + 1)
  return np.meshgrid(x, x, indexing='ij')

def nodal_grid(n):
  x = np.linspace(0.0, 1.0, n)
  return np.meshgrid(x, x, indexing='ij')

def sine_basis(n, points):
  return np.sin(np.pi * np.outer(points, np.arange(1, n + 1)))

def constant_tensor_galerkin_solution(a11, a12, a22, coefficients):
  """Exact solution for a constant tensor by assembling the sine Galerkin matrix.

  The basis is sin(k pi x) sin(l pi y) for k, l = 1..n. All the required
  one-dimensional integrals are elementary, so no spatial discretisation is
  involved: the result is what the sine Galerkin method must return, and any
  deviation from it is an error in the discretisation. coefficients[i, j] is
  the sine coefficient of f, so f is returned alongside its solution.
  """
  coefficients = np.asarray(coefficients, dtype=float)
  n = coefficients.shape[0]
  k = np.arange(1, n + 1)

  def sine_cosine(a, b):
    """The integral of sin(a pi t) cos(b pi t) over (0, 1), as an n x n array."""
    total = a[:, None] + b[None, :]
    difference = a[:, None] - b[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
      terms = (1.0 - (-1.0)**total) / (np.pi*total) + (1.0 - (-1.0)**difference) / (np.pi*difference)
    terms = np.where(np.isfinite(terms), terms, 0.0)  # the difference vanishes when a == b
    return 0.5*terms

  identity = np.eye(n)
  A, S = 0.5*identity, 0.5*identity
  X = sine_cosine(k, k)

  # B[p, q] is the bilinear form on basis function p and basis function q,
  # with p = (k1, l1) and q = (k2, l2) flattened in row-major order.
  B = np.zeros((n*n, n*n))
  for p in range(n*n):
    k1, l1 = k[p//n], k[p % n]
    for q in range(n*n):
      k2, l2 = k[q//n], k[q % n]
      B[p, q] = np.pi**2 * (
          a11*k1*k2*A[k1-1, k2-1]*S[l1-1, l2-1]
          + a22*l1*l2*S[k1-1, k2-1]*A[l1-1, l2-1]
          + a12*(k1*l2*X[k2-1, k1-1]*X[l1-1, l2-1]
                 + l1*k2*X[k1-1, k2-1]*X[l2-1, l1-1]))

  # The sine basis is orthogonal with mass 1/4 on the unit square.
  solution = np.linalg.solve(B, 0.25*coefficients.reshape(-1)).reshape(n, n)
  basis = sine_basis(n, np.arange(1, n + 1) / (n + 1))
  return basis @ solution @ basis.T, basis @ coefficients @ basis.T

class TestDarcySolver(unittest.TestCase):
  def test_constant_coefficient_is_exact_on_eigenfunction(self):
    """a = 1 with a sine eigenfunction is solved exactly by the sine Galerkin method."""
    n = 15
    X, Y = interior_grid(n)
    exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    solution = uniform_grid_solver_scalar(np.ones((n + 2, n + 2)), 2 * np.pi**2 * exact, tol=1e-12)
    self.assertLess(np.abs(solution - exact).max(), 1e-10)

  def test_constant_tensor_matches_exact_galerkin_solution(self):
    """A constant tensor with an off-diagonal entry is solved exactly.

    This is the only test here that checks the cross term rather than the
    convergence of the iteration: a solver that dropped the off-diagonal
    contribution, or extended it over the wrong grid, is off by 1e-3 here
    while both its loosest and tightest tolerances agree with each other.
    """
    coefficients = np.zeros((4, 4))
    coefficients[0, 0], coefficients[1, 2], coefficients[2, 0] = 1.0, 0.5, -0.3
    n = 4
    for a11, a12, a22 in [(2.0, 0.0, 2.0), (2.0, 1.0, 2.0), (2.0, 0.5, 2.0),
                          (2.0, -1.0, 2.0), (3.0, 1.0, 1.0), (2.0, 0.9, 1.5)]:
      exact, f = constant_tensor_galerkin_solution(a11, a12, a22, coefficients)
      a = lambda value: np.full((n + 2, n + 2), value)
      solution = uniform_grid_solver_tensor(a(a11), a(a12), a(a22), f, tol=1e-13)
      self.assertLess(np.abs(solution - exact).max(), 1e-10,
                      f"constant tensor a11={a11}, a12={a12}, a22={a22}")

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

  def test_high_contrast_tensor_with_jumping_off_diagonal(self):
    """A disc of contrast 10^4 that also jumps in the off-diagonal entry.

    The disc takes a11 = a22 = 1e3 with a12 = 0.5 inside, and a11 = a22 = 1
    with a12 = 0.1 outside, so the tensor stays positive definite across the
    jump: 1e6 - 0.25 > 0 inside and 1 - 0.01 > 0 outside.
    """
    n = 23
    XN, YN = nodal_grid(n + 2)
    radius = np.sqrt((XN - 0.5)**2 + (YN - 0.5)**2)
    inside = radius < 0.3
    a11 = np.where(inside, 1e3, 1.0)
    a12 = np.where(inside, 0.5, 0.1)
    a22 = np.where(inside, 1e3, 1.0)
    f = np.ones((n, n))
    self.assertTrue(np.all(a11 * a22 - a12**2 > 0))
    default = uniform_grid_solver_tensor(a11, a12, a22, f)
    tight = uniform_grid_solver_tensor(a11, a12, a22, f, tol=1e-11)
    self.assertLess(np.abs(default - tight).max() / np.abs(tight).max(), 1e-5)
    self.assertGreater(tight.min(), -1e-10 * np.abs(tight).max())
    # The cross term must change the answer: dropping it is not a no-op.
    no_cross = uniform_grid_solver_tensor(a11, np.zeros_like(a12), a22, f, tol=1e-11)
    self.assertGreater(np.abs(tight - no_cross).max() / np.abs(tight).max(), 1e-2)

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
