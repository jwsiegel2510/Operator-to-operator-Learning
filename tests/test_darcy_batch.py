import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from examples.data_gen import darcy_solver_2d as solver
from examples.data_gen.data_generation_darcy import generate_dataset


class TestDarcyBatch(unittest.TestCase):
    def check_solver(self, device):
        rng = np.random.default_rng(17)
        n, m = 6, 7
        a = np.ones((11, 12))
        a[3:7, 4:9] = 1e4
        f, g = rng.normal(size=(2, n, m))
        forcings = np.stack([f, 1e-7*f, 1e3*g, np.zeros_like(f)])
        prepared = solver.prepare_scalar_coefficient(a, (n, m), device=device)
        solutions = solver.uniform_grid_solver_scalar_batch(prepared, forcings, tol=1e-8)
        if device == "cuda":
            solutions = solutions.get()
        np.testing.assert_array_equal(solutions[-1], 0)
        np.testing.assert_allclose(solutions[1]/1e-7, solutions[0], rtol=2e-7, atol=1e-10)

        # Independently reconstruct the preconditioned system on this small
        # grid so the comparison also checks the relative stopping rule.
        basis = np.eye(n*m).reshape(n*m, n, m)
        P = np.column_stack([solver.apply_inverse_square_root_laplace(v).ravel() for v in basis])
        coeff = solver.extend_coefficients(a, n, m)
        zero = np.zeros_like(coeff)
        for rhs, solution in zip(forcings[:-1], solutions[:-1]):
            preconditioned = np.linalg.solve(P, solution.ravel()).reshape(n, m)
            applied = solver.apply_preconditioned_operator(coeff, zero, coeff, preconditioned)
            target = P @ rhs.ravel()
            self.assertLess(np.linalg.norm(target - applied.ravel()) / np.linalg.norm(target), 1.01e-8)

        with self.assertRaises(RuntimeError):
            solver.uniform_grid_solver_scalar_batch(prepared, forcings, maxiter=1)

    def check_dataset(self, device):
        with tempfile.TemporaryDirectory() as directory:
            rng = np.random.default_rng(31)
            with patch.object(np.random, "default_rng", return_value=rng):
                paths = generate_dataset(3, 3, a_grid=9, f_grid=5, chunk_size=2,
                                         pair_batch_size=2, output_dir=directory, device=device)
            self.assertEqual(len(paths), 2)
            for path, (start, end) in zip(paths, [(0, 2), (2, 3)]):
                data = torch.load(path, weights_only=True)
                self.assertEqual((data["operator_start"], data["operator_end"]), (start, end))
                self.assertEqual(data["a"].shape, (end-start, 9, 9))
                self.assertEqual(data["f"].shape, (end-start, 3, 5, 5))
                self.assertEqual(data["u"].shape, data["f"].shape)
                for a, fs, us in zip(data["a"], data["f"], data["u"]):
                    for f, u in zip(fs, us):
                        ref = solver.uniform_grid_solver_scalar(a.double().numpy(), f.double().numpy(), tol=1e-11)
                        relative_error = np.linalg.norm(u.numpy() - ref) / np.linalg.norm(ref)
                        self.assertLess(relative_error, 2e-4)

    def test_cpu_solver(self):
        self.check_solver("cpu")

    def test_cpu_dataset(self):
        self.check_dataset("cpu")

    def test_cuda_solver_and_dataset(self):
        try:
            import cupy
        except ImportError:
            self.skipTest("CuPy is unavailable")
        if not cupy.is_available():
            self.skipTest("No CUDA device is available")
        self.check_solver("cuda")
        self.check_dataset("cuda")
