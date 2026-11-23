"""Independent Galerkin and continuum checks for the single-side DtN map.

Run from the project root with:
    python -m unittest discover -s tests -p 'test_dirichlet_to_neumann.py'
"""

import unittest

import numpy as np

from examples.data_gen.dirichlet_to_neumann import (
    dirichlet_to_neumann_scalar,
    dirichlet_to_neumann_tensor,
)

try:
    import torch
except ImportError:
    torch = None


SIDES = ("bottom", "top", "left", "right")


def _cell_gauss(intervals, order):
    points, weights = np.polynomial.legendre.leggauss(order)
    nodes = ((np.arange(intervals)[:, None] + (points + 1) / 2) / intervals).ravel()
    weights = np.broadcast_to(weights / (2 * intervals), (intervals, order)).ravel()
    return nodes, weights


def _interpolate_bilinear(a, x, y):
    """Evaluate the nodal interpolant independently of the spectral code."""
    sx = x * (a.shape[0] - 1)
    sy = y * (a.shape[1] - 1)
    ix = np.minimum(sx.astype(int), a.shape[0] - 2)
    iy = np.minimum(sy.astype(int), a.shape[1] - 2)
    tx, ty = sx - ix, sy - iy
    return (
        a[ix[:, None], iy[None, :]] * (1-tx[:, None]) * (1-ty[None, :])
        + a[ix[:, None]+1, iy[None, :]] * tx[:, None] * (1-ty[None, :])
        + a[ix[:, None], iy[None, :]+1] * (1-tx[:, None]) * ty[None, :]
        + a[ix[:, None]+1, iy[None, :]+1] * tx[:, None] * ty[None, :]
    )


def _dense_galerkin_dtn(a11, a12, a22, n, interior_shape, side, order=18):
    """Directly integrate every entry of K, B, C using cell Gauss rules.

    This oracle uses no Fourier padding, coefficient projection, or iterative
    solve from the implementation being tested.
    """
    x, wx = _cell_gauss(a11.shape[0] - 1, order)
    y, wy = _cell_gauss(a11.shape[1] - 1, order)
    x, y = x[:, None], y[None, :]
    gx, gy = [], []
    nx, ny = interior_shape
    for p in range(1, nx + 1):
        for q in range(1, ny + 1):
            gx.append(p*np.pi*np.cos(p*np.pi*x)*np.sin(q*np.pi*y))
            gy.append(q*np.pi*np.sin(p*np.pi*x)*np.cos(q*np.pi*y))
    for k in range(1, n + 1):
        if side in ("bottom", "top"):
            linear = 1-y if side == "bottom" else y
            derivative = -1 if side == "bottom" else 1
            gx.append(k*np.pi*np.cos(k*np.pi*x)*linear)
            gy.append(np.sin(k*np.pi*x)*np.ones_like(y)*derivative)
        else:
            linear = 1-x if side == "left" else x
            derivative = -1 if side == "left" else 1
            gx.append(np.ones_like(x)*np.sin(k*np.pi*y)*derivative)
            gy.append(linear*k*np.pi*np.cos(k*np.pi*y))
    gx, gy = np.stack(gx), np.stack(gy)
    a11q = _interpolate_bilinear(a11, x[:, 0], y[0])
    a12q = _interpolate_bilinear(a12, x[:, 0], y[0])
    a22q = _interpolate_bilinear(a22, x[:, 0], y[0])
    weights = wx[:, None]*wy[None, :]
    energy = np.einsum("ixy,jxy,xy->ij", gx, gx, weights*a11q)
    energy += np.einsum("ixy,jxy,xy->ij", gx, gy, weights*a12q)
    energy += np.einsum("ixy,jxy,xy->ij", gy, gx, weights*a12q)
    energy += np.einsum("ixy,jxy,xy->ij", gy, gy, weights*a22q)
    size = nx*ny
    stiffness = energy[:size, :size]
    coupling = energy[:size, size:]
    lift_energy = energy[size:, size:]
    # The boundary sine mass matrix is (1/2) I.
    return 2*(lift_energy - coupling.T @ np.linalg.solve(stiffness, coupling))


def _constant_diagonal_dtn(a_tangent, a_normal, n, normal_modes):
    """Exact finite sine-Galerkin Schur complement for constant diagonal A."""
    k = np.arange(1, n+1, dtype=float)
    q = np.arange(1, normal_modes+1, dtype=float)
    diagonal = a_tangent*(np.pi*k)**2/3 + a_normal
    diagonal -= np.sum(
        2*a_tangent**2*k[:, None]**4
        / (q[None, :]**2*(a_tangent*k[:, None]**2 + a_normal*q[None, :]**2)),
        axis=1,
    )
    return np.diag(diagonal)


class DirichletToNeumannTests(unittest.TestCase):
    def setUp(self):
        # The coefficient is bilinear on each cell and SPD at all nodes.
        rng = np.random.default_rng(714)
        self.a11 = 2 + 0.4*rng.random((3, 4))
        self.a12 = 0.25 + 0.15*rng.random((3, 4))
        self.a22 = 1.1 + 0.3*rng.random((3, 4))

    def test_constant_diagonal_exact_finite_galerkin_all_sides(self):
        nx, ny, n = 5, 7, 3
        a11, a22 = 1.7, 0.8
        for side in SIDES:
            with self.subTest(side=side):
                actual = dirichlet_to_neumann_tensor(
                    np.full((3, 4), a11), np.zeros((3, 4)),
                    np.full((3, 4), a22), n, side=side,
                    interior_shape=(nx, ny), tol=1e-12,
                )
                if side in ("bottom", "top"):
                    expected = _constant_diagonal_dtn(a11, a22, n, ny)
                else:
                    expected = _constant_diagonal_dtn(a22, a11, n, nx)
                np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)

    def test_constant_scalar_converges_to_continuum_map(self):
        n, a = 3, 1.2
        k = np.arange(1, n+1)*np.pi
        expected = np.diag(a*k/np.tanh(k))
        coarse = dirichlet_to_neumann_scalar(
            np.full((2, 2), a), n, interior_shape=(8, 8), tol=1e-12,
        )
        fine = dirichlet_to_neumann_scalar(
            np.full((2, 2), a), n, interior_shape=(64, 64), tol=1e-12,
        )
        self.assertLess(np.linalg.norm(fine-expected), np.linalg.norm(coarse-expected)/100)
        np.testing.assert_allclose(fine, expected, rtol=0, atol=3e-4)

    def test_variable_tensor_matches_independent_dense_assembly(self):
        for side in SIDES:
            with self.subTest(side=side):
                actual = dirichlet_to_neumann_tensor(
                    self.a11, self.a12, self.a22, 3,
                    interior_shape=(4, 3), side=side, tol=1e-12,
                )
                expected = _dense_galerkin_dtn(
                    self.a11, self.a12, self.a22, 3, (4, 3), side,
                )
                np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
                np.testing.assert_allclose(actual, actual.T, rtol=0, atol=1e-10)
                self.assertGreater(np.linalg.eigvalsh(actual).min(), 0)

    def test_boundary_resolution_can_exceed_interior_tangential_resolution(self):
        a11, a22 = np.full((2, 3), 1.4), np.full((2, 3), 0.9)
        a12 = np.zeros_like(a11)
        # Some boundary modes have no matching interior tangential mode.
        # Their energy still contributes through the auxiliary lifting space.
        for side in SIDES:
            with self.subTest(side=side):
                actual = dirichlet_to_neumann_tensor(
                    a11, a12, a22, 4, interior_shape=(2, 3),
                    side=side, tol=1e-12,
                )
                expected = _dense_galerkin_dtn(a11, a12, a22, 4, (2, 3), side)
                np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)

    def test_minimal_interior_and_coefficient_grids_with_cross_diffusion(self):
        a11 = np.array([[1.8, 2.1], [2.4, 2.0]])
        a12 = np.array([[0.3, 0.2], [0.4, -0.1]])
        a22 = np.array([[1.2, 1.4], [1.3, 1.1]])
        for side in SIDES:
            with self.subTest(side=side):
                actual = dirichlet_to_neumann_tensor(
                    a11, a12, a22, 2, interior_shape=(1, 1), side=side, tol=1e-12,
                )
                expected = _dense_galerkin_dtn(a11, a12, a22, 2, (1, 1), side)
                np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)

    def test_explicit_quadrature_order_converges(self):
        a11 = np.array([[2.1, 2.3], [2.4, 1.9]])
        a12 = np.array([[0.2, -0.1], [0.3, 0.4]])
        a22 = np.array([[1.1, 1.4], [1.3, 1.2]])
        expected = _dense_galerkin_dtn(a11, a12, a22, 3, (3, 3), "bottom", order=28)
        low = dirichlet_to_neumann_tensor(
            a11, a12, a22, 3, quadrature_order=4, tol=1e-12,
        )
        high = dirichlet_to_neumann_tensor(
            a11, a12, a22, 3, quadrature_order=16, tol=1e-12,
        )
        self.assertLess(np.linalg.norm(high-expected), np.linalg.norm(low-expected)/1000)
        np.testing.assert_allclose(high, expected, rtol=1e-10, atol=1e-10)

    def test_nonconvergence_raises(self):
        with self.assertRaises(RuntimeError):
            dirichlet_to_neumann_tensor(
                self.a11, self.a12, self.a22, 3, interior_shape=(5, 4),
                maxiter=1, tol=1e-13,
            )

    def test_scalar_and_diagonal_tensor_agree(self):
        scalar = dirichlet_to_neumann_scalar(self.a11, 3, tol=1e-12)
        tensor = dirichlet_to_neumann_tensor(
            self.a11, np.zeros_like(self.a11), self.a11, 3, tol=1e-12,
        )
        np.testing.assert_allclose(scalar, tensor, rtol=2e-12, atol=2e-12)

    def test_invalid_inputs(self):
        a = np.ones((3, 4))
        for kwargs in ({"side": "unknown"}, {"interior_shape": (0, 3)}, {"tol": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                dirichlet_to_neumann_scalar(a, 2, **kwargs)
        for n in (0, -1):
            with self.subTest(n=n), self.assertRaises(ValueError):
                dirichlet_to_neumann_scalar(a, n)
        with self.assertRaises(ValueError):
            dirichlet_to_neumann_scalar(np.zeros((3, 4)), 2)
        with self.assertRaises(ValueError):
            dirichlet_to_neumann_tensor(a, 2*a, a, 2)

    def test_optional_solver_information(self):
        matrix, information = dirichlet_to_neumann_scalar(
            np.ones((2, 2)), 2, return_info=True,
        )
        self.assertEqual(matrix.shape, (2, 2))
        self.assertIsInstance(information, dict)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_torch_cpu_agrees_with_numpy(self):
        expected = dirichlet_to_neumann_tensor(
            self.a11, self.a12, self.a22, 3, interior_shape=(4, 3), tol=1e-11,
        )
        actual = dirichlet_to_neumann_tensor(
            torch.tensor(self.a11), torch.tensor(self.a12), torch.tensor(self.a22),
            3, interior_shape=(4, 3), tol=1e-11,
        )
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(actual.dtype, torch.float64)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-10, atol=2e-10)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_torch_float32_agrees_with_float64_reference(self):
        expected = _dense_galerkin_dtn(
            self.a11, self.a12, self.a22, 3, (4, 3), "bottom",
        )
        actual = dirichlet_to_neumann_tensor(
            torch.tensor(self.a11, dtype=torch.float32),
            torch.tensor(self.a12, dtype=torch.float32),
            torch.tensor(self.a22, dtype=torch.float32),
            3, interior_shape=(4, 3), tol=2e-5,
        )
        self.assertEqual(actual.dtype, torch.float32)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-4, atol=2e-4)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_agrees_with_numpy(self):
        expected = dirichlet_to_neumann_tensor(
            self.a11, self.a12, self.a22, 3, interior_shape=(4, 3), tol=1e-10,
        )
        actual = dirichlet_to_neumann_tensor(
            self.a11, self.a12, self.a22, 3, interior_shape=(4, 3),
            device="cuda", dtype=torch.float64, tol=1e-10,
        )
        self.assertEqual(actual.device.type, "cuda")
        np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
