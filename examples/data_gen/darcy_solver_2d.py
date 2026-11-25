# Author: Jonathan Siegel
#
# Implements solvers for the 2d darcy problem with tensor diffusion coefficient.

from collections import namedtuple

import numpy as np
from math import sqrt

def uniform_grid_solver_scalar(a, f, verbose=False, tol=10e-6, maxiter=None):
  """Solves the Darcy equation with scalar diffusion a and forcing term f with zero boundary conditions.
     The inputs a and f are assumed to be given on a uniform grid on the square. f is taken to be a 
     trigonometric sin polynomial given by its values at a uniform grid, and a is given by its values
     on a (potentially different) uniform grid via piecewise bilinear interpolation.

  args:
    a: scalar diffusion coefficient
    f: forcing term
    verbose: print per-iteration conjugate gradient diagnostics
    tol: relative tolerance on the RMS of the preconditioned residual
    maxiter: conjugate gradient iteration limit

  returns:
    the solution (given on a grid of the same dimensions as f).
  """
  return uniform_grid_solver_tensor(a, np.zeros_like(a), a, f, tol, verbose=verbose, maxiter=maxiter, diagonal_diffusion=True, scalar_diffusion=True)

# All arrays in a prepared coefficient live on the selected device and are
# reused across batches of forcing fields for this operator.
_PreparedScalarCoefficient = namedtuple(
    "_PreparedScalarCoefficient", "xp values inverse_laplace_root grad_x grad_y"
)


def prepare_scalar_coefficient(a, grid_shape, *, device="cpu"):
  """Prepare a positive nodal coefficient for an (n, m) interior solution grid."""
  if device == "cpu":
    xp = np
  elif device == "cuda":
    try:
      import cupy as xp
    except ImportError as exc:
      raise RuntimeError("CUDA solves require CuPy matching the installed CUDA toolkit") from exc
    if not xp.is_available():
      raise RuntimeError("CuPy could not find an available CUDA device")
  else:
    raise ValueError("device must be 'cpu' or 'cuda'")

  n, m = grid_shape
  a = xp.asarray(a, dtype=xp.float64)
  if a.ndim != 2 or min(a.shape) < 2 or min(n, m) < 1:
    raise ValueError("a needs at least two nodes per axis and the interior grid must be nonempty")
  if not bool(xp.all(xp.isfinite(a) & (a > 0))):
    raise ValueError("Scalar diffusion coefficients must be finite and positive")
  values = extend_coefficients(a, n, m, xp=xp)
  kx = xp.arange(1, n+1, dtype=xp.float64)[:, None]
  ky = xp.arange(1, m+1, dtype=xp.float64)[None, :]
  radius = xp.hypot(kx, ky)
  return _PreparedScalarCoefficient(xp, values, 1 / (np.pi * radius), kx / radius, ky / radius)


def _modes_to_grid(modes, shape, xp, *, even_x=False):
  """Invert the positive sine modes, or the corresponding gradient modes."""
  n, m = modes.shape[-2:]
  # rfft2 stores only nonnegative y frequencies. Reflection in x is odd
  # for a sine series, and even after taking its x derivative.
  spectrum = xp.zeros((*modes.shape[:-2], shape[0], shape[1]//2 + 1), dtype=xp.complex128)
  spectrum[..., 1:n+1, 1:m+1] = modes
  reflected = modes[..., ::-1, :]
  spectrum[..., -n:, 1:m+1] = reflected if even_x else -reflected
  return xp.fft.irfft2(spectrum, s=shape, axes=(-2, -1), norm="forward")


def _apply_scalar_modes(prepared, modes):
  """Apply G* a G in sine space, where G = grad(-Δ)^(-1/2)."""
  xp = prepared.xp
  n, m = modes.shape[-2:]
  result = xp.zeros_like(modes)
  # The two gradient multipliers are i*kx/|k| and i*ky/|k|. Multiply by
  # the projected coefficient on the same enlarged grid as the tensor solver.
  for weight, even_x in ((prepared.grad_x, True), (prepared.grad_y, False)):
    flux = _modes_to_grid(1j * weight * modes, prepared.values.shape, xp, even_x=even_x)
    flux *= prepared.values
    flux_modes = xp.fft.rfft2(flux, axes=(-2, -1), norm="forward")
    result += weight * flux_modes[..., 1:n+1, 1:m+1].imag
  return result


def uniform_grid_solver_scalar_batch(prepared, f, tol=1e-5, maxiter=None):
  """Solve f[batch, n, m] in float64 on the prepared coefficient's device.

  Returns a NumPy (CPU) or CuPy (CUDA) array. Each pair has its own relative
  preconditioned RMS residual target and the same iteration limit as the
  single-forcing solver. Zero forcings give zero solutions.
  """
  xp = prepared.xp
  f = xp.asarray(f, dtype=xp.float64)
  if f.ndim != 3 or f.shape[0] == 0 or f.shape[-2:] != prepared.grad_x.shape:
    raise ValueError("f must contain a nonempty batch on the prepared interior grid")
  if not bool(xp.all(xp.isfinite(f))):
    raise ValueError("f must be finite")
  if not np.isfinite(tol) or tol <= 0:
    raise ValueError("tol must be a positive finite number")
  _, n, m = f.shape
  if maxiter is None:
    maxiter = min(10*n*m, 10000)

  extended = extend_antisymmetrically(f, xp=xp)
  spectrum = xp.fft.rfft2(extended, axes=(-2, -1), norm="forward")
  residual = spectrum[..., 1:n+1, 1:m+1].real * prepared.inverse_laplace_root
  del extended, spectrum
  solution = xp.zeros_like(residual)
  direction = residual.copy()
  # By Parseval, the spatial and sine-mode squared norms differ by a
  # constant that cancels in each RHS's relative convergence criterion.
  rr = xp.sum(residual * residual, axis=(-2, -1), keepdims=True)
  target = tol**2 * rr
  active = rr > target

  for _ in range(maxiter):
    if not bool(active.any()):
      break
    applied = _apply_scalar_modes(prepared, direction)
    curvature = xp.sum(direction * applied, axis=(-2, -1), keepdims=True)
    # A bad active system propagates NaN to the final convergence check.
    # Inactive systems take a zero step and never divide by zero.
    curvature = xp.where((curvature > 0) & xp.isfinite(curvature), curvature, xp.nan)
    step = xp.where(active, rr, 0) / xp.where(active, curvature, 1)
    solution += step * direction
    residual -= step * applied
    rr_new = xp.sum(residual * residual, axis=(-2, -1), keepdims=True)
    active = rr_new > target
    beta = xp.where(active, rr_new, 0) / xp.where(active, rr, 1)
    direction = xp.where(active, residual + beta * direction, 0)
    rr = rr_new

  if not bool(xp.all(xp.isfinite(rr) & (rr <= target))):
    raise RuntimeError(f"Scalar Darcy CG broke down or did not converge in {maxiter} iterations")
  values = _modes_to_grid(solution * prepared.inverse_laplace_root,
                          (2*n + 2, 2*m + 2), xp)
  return values[..., 1:n+1, 1:m+1]


def uniform_grid_solver_tensor(a11, a12, a22, f, tol=10e-6, verbose=False, maxiter=None, diagonal_diffusion=False, scalar_diffusion=False):
  """Solves the Darcy equation with tensor diffusion a and forcing term f with zero boundary conditions.
     The inputs a and f are assumed to be given on a uniform grid on the square. f is taken to be a 
     trigonometric sin polynomial given by its values at a uniform grid, and a is given by its values
     on a (potentially different) uniform grid via piecewise bilinear interpolation.

  args:
    a11, a12, a22: tensor diffusion coefficient
    f: forcing term
    tol: relative tolerance on the RMS of the preconditioned residual, measured
         against its initial value, so that the result does not depend on the
         scaling of f
    verbose: print per-iteration conjugate gradient diagnostics
    maxiter: conjugate gradient iteration limit; defaults to 10*n*m capped at 10000

  returns:
    the solution (given on a grid of the same dimensions as f).

  raises:
    ValueError: if an input is nonfinite, has an unexpected shape, or the
      diffusion tensor is not positive definite
    RuntimeError: if conjugate gradient breaks down or does not reach the tolerance
  """
  # Preprocess the diffusion coefficients a. Calculate their projections onto a large enough trigonometric
  # polynomial space and extend to a large enough grid to allow multiplication by elements of the Galerkin space.
  if np.ndim(f) != 2 or min(np.shape(f)) < 1:
    raise ValueError("f must be a nonempty 2D interior grid")
  n, m = np.shape(f)
  if np.shape(a11) != np.shape(a22) or np.shape(a12) != np.shape(a11):
    raise ValueError("Coefficient arrays must share a single 2D shape")
  for name, a in (("a11", a11), ("a12", a12), ("a22", a22), ("f", f)):
    if not np.all(np.isfinite(a)):
      raise ValueError(f"{name} must be finite")
  if np.any(a11 <= 0) or np.any(a22 <= 0):
    raise ValueError("The diagonal of the diffusion tensor must be positive")
  if np.any(a11 * a22 - a12 * a12 <= 0):
    raise ValueError("The diffusion tensor must be positive definite")
  if not np.isfinite(tol) or tol <= 0:
    raise ValueError("tol must be a positive finite number")
  if maxiter is None:
    maxiter = min(10*n*m, 10000)
  a11 = extend_coefficients(a11, n, m)
  if diagonal_diffusion:
    a12 = np.zeros_like(a11)
  else:
    a12 = extend_coefficients(a12, n, m, off_diag = True)
  if scalar_diffusion:
    a22 = a11
  else:
    a22 = extend_coefficients(a22, n, m)

  # Apply square root laplace preconditioner to f.
  f = apply_inverse_square_root_laplace(f)

  # Implement CG on the preconditioned operator.
  u = np.zeros_like(f)
  r = f
  p = r
  rr = np.mean(r * r)
  target = tol * sqrt(rr)
  iter = 1
  while (sqrt(rr) > target and iter <= maxiter):
    Ap = apply_preconditioned_operator(a11, a12, a22, p, diagonal_diffusion=diagonal_diffusion)
    denominator = np.mean(p * Ap)
    if not np.isfinite(denominator) or denominator <= 0:
      raise RuntimeError("CG breakdown: nonfinite residual or nonpositive curvature")
    alpha = rr / denominator
    u = u + alpha * p
    r_new = r - alpha * Ap
    rr_new = np.mean(r_new * r_new)
    if not np.isfinite(rr_new):
      raise RuntimeError("CG residual became nonfinite")
    if verbose:
      Ar = apply_preconditioned_operator(a11, a12, a22, r, diagonal_diffusion=diagonal_diffusion)
      print(iter)
      print(np.mean(p * Ar), np.mean(r * Ap))
      print(np.mean(p * Ap))
    r = r_new
    p = r + (rr_new / rr) * p
    rr = rr_new
    iter = iter + 1
  if sqrt(rr) > target:
    raise RuntimeError(f"CG did not converge in {maxiter} iterations (preconditioned residual RMS={sqrt(rr):.3e}, target={target:.3e})")

  # Apply square root laplace preconditioner to the solution.
  return apply_inverse_square_root_laplace(u)

def _next_power_of_two(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def apply_preconditioned_operator(a11, a12, a22, u, diagonal_diffusion=False):
  """Applies the Darcy operator, symmetrically preconditioned by the Laplacian. The coefficients a are
     assumed to already be extended to a large enough grid by the function extend_coefficients below.

  args:
    a11, a12, a22: tensor diffusion coefficient
    u: iterate
    diagonal_diffusion: tells whether the off-diagonal terms are zero (and their multiplications can be saved)

  returns:
    next iterate
  """
  n, m = np.shape(u)

  # Apply right preconditioner matrix.
  u1 = apply_inverse_square_root_laplace(u)

  # Extend u1 antisymmetrically.
  extended_grid_u1 = extend_antisymmetrically(u1)

  # Evaluate the Fourier transform and extend to a large enough grid to allow multiplication.
  fft_u1 = np.fft.fft2(extended_grid_u1, norm='forward')

  large_n = _next_power_of_two(4*n + 7)
  large_m = _next_power_of_two(4*m + 7)

  fft_u1 = extend_fft(fft_u1, large_n, large_m)

  # Calculate the gradient of the extended u1 and move to real space.
  x_factors = np.pi * 1.0j * np.minimum(np.arange(large_n), np.arange(large_n,0,-1)) * np.where(np.arange(large_n) > np.arange(large_n,0,-1), -1.0, 1.0)
  y_factors = np.pi * 1.0j * np.minimum(np.arange(large_m), np.arange(large_m,0,-1)) * np.where(np.arange(large_m) > np.arange(large_m,0,-1), -1.0, 1.0)
  factx, facty = np.meshgrid(x_factors, y_factors, indexing='ij') 
  grad_u1_x = np.real(np.fft.ifft2(fft_u1 * factx, norm='forward'))
  grad_u1_y = np.real(np.fft.ifft2(fft_u1 * facty, norm='forward'))

  # Multiply the gradients with the coefficients a in real space.
  if diagonal_diffusion:
    out_grad_u1_x = a11 * grad_u1_x 
    out_grad_u1_y = a22 * grad_u1_y
  else:
    out_grad_u1_x = a11 * grad_u1_x + a12 * grad_u1_y 
    out_grad_u1_y = a12 * grad_u1_x + a22 * grad_u1_y
  
  # Return to Fourier space and calculate the negative divergence.
  new_fft_x = np.fft.fft2(out_grad_u1_x, norm='forward')
  new_fft_y = np.fft.fft2(out_grad_u1_y, norm='forward')
  out_fft = -1.0 * (factx * new_fft_x + facty * new_fft_y)

  # Project back onto the original space of trigonometric polynomials.
  out_fft = project_fft(out_fft, 2*n + 2, 2*m + 2)
  
  # Project onto the sin space (of antisymmetric functions on the larger grid).
  extended_out_u = np.real(np.fft.ifft2(out_fft, norm='forward'))
  out_u = (1.0/4.0)*(extended_out_u[1:n+1, 1:m+1] + extended_out_u[:n+1:-1, :m+1:-1] - extended_out_u[1:n+1, :m+1:-1] - extended_out_u[:n+1:-1, 1:m+1])

  # Apply other half of Laplace preconditioner.
  return apply_inverse_square_root_laplace(out_u)

def extend_coefficients(a, n, m, off_diag=False, *, xp=np):
  """Project a piecewise bilinear coefficient onto the padded product grid."""
  if off_diag:
    extended = extend_antisymmetrically(a[1:-1, 1:-1], xp=xp)
  else:
    extended = extend_symmetrically(a, xp=xp)
  ax, ay = extended.shape
  transform = xp.fft.fft2(extended, norm="forward")
  # Fourier multiplier of the bilinear nodal basis, through the original
  # Galerkin cutoff of 2*n+3 and 2*m+3 on each side of zero.
  kx = xp.arange(-2*n - 3, 2*n + 4)[:, None]
  ky = xp.arange(-2*m - 3, 2*m + 4)[None, :]
  weights = xp.sinc(kx / ax)**2 * xp.sinc(ky / ay)**2
  shape = (_next_power_of_two(4*n + 7), _next_power_of_two(4*m + 7))
  spectrum = xp.zeros(shape, dtype=xp.complex128)
  spectrum[kx, ky] = weights * transform[kx % ax, ky % ay]
  return xp.ascontiguousarray(xp.fft.ifft2(spectrum, norm="forward").real)


def extend_symmetrically(a, *, xp=np):
  """Reflect the coefficient evenly about both boundary axes."""
  a = xp.asarray(a, dtype=xp.float64)
  reflected = xp.concatenate((a, a[-2:0:-1, :]), axis=0)
  return xp.concatenate((reflected, reflected[:, -2:0:-1]), axis=1)


def extend_antisymmetrically(u, *, xp=np):
  """Reflect the last two axes oddly across the zero Dirichlet boundaries."""
  n, m = u.shape[-2:]
  extended = xp.zeros((*u.shape[:-2], 2*n + 2, 2*m + 2), dtype=xp.float64)
  extended[..., 1:n+1, 1:m+1] = u
  extended[..., n+2:, 1:m+1] = -u[..., ::-1, :]
  extended[..., 1:n+1, m+2:] = -u[..., :, ::-1]
  extended[..., n+2:, m+2:] = u[..., ::-1, ::-1]
  return extended


def extend_fft(u_fft, n, m):
  """Given the Fourier transform of a function u, extends this to a larger grid with sizes n x m.
     If n or m are even, we must take care to split the half frequencies into positive and negative parts.

  args:
    u_fft: input Fourier transform
    n, m: dimensions of the larger grid, should both be at least as large as the dimensions of u

  returns:
    extended Fourier transform on a grid of size n x m.
  """
  extended_fft = np.zeros((n, m), dtype=complex)
  n_u, m_u = np.shape(u_fft)

  # Determine largest odd numbers less than n_u and m_u.
  n_odd = 2 * ((n_u-1)//2) + 1
  m_odd = 2 * ((m_u-1)//2) + 1

  # Fill in all of the Fourier modes except the potential middle frequency.
  extended_fft[0:n_odd // 2 + 1, 0:m_odd // 2 + 1] += u_fft[0:n_odd // 2 + 1, 0:m_odd // 2 + 1]
  extended_fft[0:n_odd // 2 + 1, :-(m_odd // 2)-1:-1] += u_fft[0:n_odd // 2 + 1, :-(m_odd // 2)-1:-1]
  extended_fft[:-(n_odd // 2)-1:-1, 0:m_odd // 2 + 1] += u_fft[:-(n_odd // 2)-1:-1, 0:m_odd // 2 + 1]
  extended_fft[:-(n_odd // 2)-1:-1, :-(m_odd // 2)-1:-1] += u_fft[:-(n_odd // 2)-1:-1, :-(m_odd // 2)-1:-1]

  # Fill in the remaining frequencies. Half frequencies must be split between positive and negative pieces.
  if n_u % 2 == 0:
    extended_fft[n_u // 2, 0:m_odd // 2 + 1] += 0.5 * u_fft[n_u // 2, 0:m_odd // 2 + 1]
    extended_fft[-(n_u // 2), 0:m_odd // 2 + 1] += 0.5 * u_fft[n_u // 2, 0:m_odd // 2 + 1]

  if m_u % 2 == 0:
    extended_fft[0:n_odd // 2 + 1, m_u // 2] += 0.5 * u_fft[0:n_odd // 2 + 1, m_u // 2]
    extended_fft[0:n_odd // 2 + 1, -(m_u // 2)] = 0.5 * u_fft[0:n_odd // 2 + 1, m_u // 2]

  if (n_u % 2 == 0) and (m_u//2 == 0):
    extended_fft[n_u // 2, m_u // 2] += 0.25 * u_fft[n_u // 2, m_u // 2]
    extended_fft[-(n_u // 2), m_u // 2] += 0.25 * u_fft[n_u // 2, m_u // 2]
    extended_fft[n_u // 2, -(m_u // 2)] += 0.25 * u_fft[n_u // 2, m_u // 2]
    extended_fft[-(n_u // 2), -(m_u // 2)] += 0.25 * u_fft[n_u // 2, m_u // 2]

  return extended_fft

def project_fft(u_fft, n, m):
  """Projects the Fourier transform of a function u onto the space spanned by n x m Fourier frequencies.

  args:
    u_fft: input Fourier transform
    n, m: dimensions of the projected grid, should both be at most as large as the dimensions of u

  returns:
    Projected Fourier transform on a grid of size n x m.
  """
  projected_fft = np.zeros((n, m), dtype=complex)
  
  # Determine the smallest odd numbers larger than n and m.
  n_odd = 2 * (n // 2) + 1
  m_odd = 2 * (m // 2) + 1

  # Fill in the frequencies. For even n or m, the middle frequency is hit twice.
  projected_fft[0:n_odd // 2 + 1, 0:m_odd // 2 + 1] += u_fft[0:n_odd // 2 + 1, 0:m_odd // 2 + 1]
  projected_fft[0:n_odd // 2 + 1, :-(m_odd // 2)-1:-1] += u_fft[0:n_odd // 2 + 1, :-(m_odd // 2)-1:-1]
  projected_fft[:-(n_odd // 2)-1:-1, 0:m_odd // 2 + 1] += u_fft[:-(n_odd // 2)-1:-1, 0:m_odd // 2 + 1]
  projected_fft[:-(n_odd // 2)-1:-1, :-(m_odd // 2)-1:-1] += u_fft[:-(n_odd // 2)-1:-1, :-(m_odd // 2)-1:-1]

  return projected_fft

def apply_inverse_square_root_laplace(u):
  """Applies the inverse of the square root of the Laplacian with zero-boundary conditions on a uniform grid.

  args:
    u: input

  returns:
    square root of the inverse Laplace operator applied to u.
  """
  n, m = np.shape(u)

  # Extend u antisymmetrically using the zero boundary conditions.
  extended_grid = extend_antisymmetrically(u)
  
  # Evaluate the Fourier transform and apply the appropriate multiplier. Since the multiplier is symmetric
  # no extension to an odd grid is necessary.
  fourier_transform = np.fft.fft2(extended_grid)
  x_factors = np.minimum(np.arange(2*n+2), np.arange(2*n+2,0,-1))
  y_factors = np.minimum(np.arange(2*m+2), np.arange(2*m+2,0,-1))
  xf, yf = np.meshgrid(x_factors, y_factors, indexing='ij')
  scaling_factors = np.pi * np.sqrt(xf * xf + yf * yf)
  fourier_transform[1:, 1:] = fourier_transform[1:, 1:] / scaling_factors[1:, 1:]

  # Invert the Fourier transform and extract the function.
  inverse_transform = np.fft.ifft2(fourier_transform)
  return np.real(inverse_transform[1:n+1, 1:m+1])

if __name__ == '__main__':
  n, m = (512, 512)
  f = np.ones((n, m))
  a = np.ones((3*n, 3*m))
  a[n:2*n, m:2*m] = 100
  u = uniform_grid_solver_scalar(a, f)
