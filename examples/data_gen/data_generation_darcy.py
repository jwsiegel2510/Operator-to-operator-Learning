"""Generate scalar Darcy training pairs on the unit square, grouped by coefficient."""

import argparse
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .darcy_solver_2d import uniform_grid_solver_scalar
    from .random_sampling import sample_gaussian_field, sample_scalar_diffusion
else:  # Also support running this file directly.
    from darcy_solver_2d import uniform_grid_solver_scalar
    from random_sampling import sample_gaussian_field, sample_scalar_diffusion


def generate_dataset(no_of_operators, no_of_examples=40, *, a_grid=65, f_grid=32,
                     chunk_size=8, output_dir="darcy_data"):
    """Save float32 .pt chunks and return their paths.

    For C operators in a chunk, a has shape (C, a_grid, a_grid), and f and u
    have shape (C, no_of_examples, f_grid, f_grid). Global operator indices
    run from operator_start (inclusive) to operator_end (exclusive).
    """
    if a_grid < 2 or min(f_grid, no_of_operators, no_of_examples, chunk_size) < 1:
        raise ValueError("a_grid must be at least 2, and all other sizes must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("darcy_chunk_*.pt")):
        raise FileExistsError(f"{output_dir} already contains Darcy chunks; choose a fresh output directory")

    paths = []
    for start in range(0, no_of_operators, chunk_size):
        end = min(start + chunk_size, no_of_operators)
        a = np.empty((end - start, a_grid, a_grid), dtype=np.float32)
        f = np.empty((end - start, no_of_examples, f_grid, f_grid), dtype=np.float32)
        u = np.empty_like(f)

        for i in range(end - start):
            a[i] = sample_scalar_diffusion(a_grid, a_grid)
            # Solve in float64 using the same coefficient and forcing values we save.
            coefficient = a[i].astype(np.float64)
            for j in range(no_of_examples):
                f[i, j] = 2.0 * sample_gaussian_field(f_grid, f_grid, width=0.4)
                u[i, j] = uniform_grid_solver_scalar(coefficient, f[i, j].astype(np.float64))

        path = output_dir / f"darcy_chunk_{start:06d}_{end:06d}.pt"
        partial = path.with_suffix(".pt.partial")
        with partial.open("xb") as stream:
            torch.save({"a": torch.from_numpy(a), "f": torch.from_numpy(f),
                        "u": torch.from_numpy(u), "operator_start": start,
                        "operator_end": end}, stream)
        partial.replace(path)
        paths.append(path)
        print(f"Saved {path}  a={a.shape}  f={f.shape}  u={u.shape}", flush=True)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-of-operators", type=int, required=True)
    parser.add_argument("--no-of-examples", type=int, default=40, help="Forcing–solution pairs per operator")
    parser.add_argument("--a-grid", type=int, default=65, help="Coefficient nodes per axis, including boundaries")
    parser.add_argument("--f-grid", type=int, default=32, help="Interior nodes per axis for f and u")
    parser.add_argument("--chunk-size", type=int, default=8, help="Operators per .pt file")
    parser.add_argument("--output-dir", type=Path, default=Path("darcy_data"))
    generate_dataset(**vars(parser.parse_args()))
