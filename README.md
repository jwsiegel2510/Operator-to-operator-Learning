# Operator-to-operator Learning

Contains experiments testing and demonstrating the operator-to-operator learning methodology. Currently under construction.

## Single-side Dirichlet-to-Neumann map

The spectral solver works on the unit square, with array axes `(x, y)`. Build
an `n x n` map from sine coefficients of boundary values to sine coefficients
of the **outward conormal derivative** `nu . A grad(u)` on the same side:

```python
import numpy as np
from examples.data_gen import dirichlet_to_neumann_scalar

a = np.ones((33, 33))  # nodal diffusion, including boundary nodes
D, info = dirichlet_to_neumann_scalar(
    a, n=8, side="bottom", interior_shape=(32, 32),
    tol=1e-10, return_info=True,
)
g = np.zeros(8)
g[0] = 1.0
flux_coefficients = D @ g
```

The boundary basis is `sin(k*pi*t)`, `k=1,...,n`, because the side length is one.
Here `t=x` on bottom/top and `t=y` on left/right; it always increases with the
physical coordinate. `side` is `"bottom"` (`y=0`), `"top"` (`y=1`), `"left"`
(`x=0`), or `"right"` (`x=1`). The other three sides have zero Dirichlet data.
To evaluate the projected flux at side positions `t`, use
`np.sin(np.pi * t[:, None] * np.arange(1, 9)) @ flux_coefficients`.
The sign is that of the outward conormal derivative, rather than Darcy velocity.

For full symmetric tensor diffusion, use
`dirichlet_to_neumann_tensor(a11, a12, a22, n, ...)` with the same options.
All coefficient arrays specify a piecewise bilinear interpolant on the same
grid and must give a positive definite tensor at every node.

The auxiliary lift of each boundary mode is `sin(k*pi*t)*L(s)`, where `s` is
the other coordinate: `L=1-s` on bottom/left, and `L=s` on top/right. With
interior sine-sine stiffness matrix `K`, interior/lift energy coupling `B`,
and lift energy matrix `C`, the returned coefficient matrix is
`D = 2*(C - B.T @ solve(K, B))`. The factor two accounts for the boundary sine
mass `1/2`. Interior corrections reuse the spectral operator and preconditioned
CG; the dense interior stiffness matrix is never assembled. This weak flux
preserves the Galerkin energy pairing; it is not a pointwise boundary derivative
of the truncated solution.

`interior_shape=(nx, ny)` sets the number of interior modes in each coordinate;
it defaults to `(n, n)`. Refine it to check the interior approximation error.
For constant scalar diffusion `a`, the continuum map is diagonal with entries
`a*k*pi/tanh(k*pi)`; the finite interior approximation approaches these values.
Lift couplings use composite Gauss integration of separable one-dimensional
moments on coefficient cells. The default order is chosen from the frequency
per cell; an explicit `quadrature_order>=2` can be supplied for convergence
checks. `tol` controls the preconditioned CG residual, separately from these
approximation errors. Coupling storage scales as `n*nx*ny`, and building the
full map requires one interior solve per boundary mode.

NumPy inputs use the CPU. Existing PyTorch tensors retain their device; pass
`device="cuda"` to select NVIDIA GPUs, and optionally `dtype=torch.float32`
with `tol=1e-5` (the default `tol=1e-10` is intended for float64). The default
for NumPy inputs is float64.
`return_info=True` returns the matrix and a dictionary containing `side`,
`interior_shape`, and per-column CG information under `solves`.

Run numerical checks from the project root:

```sh
python -m unittest discover -s tests -v
```
