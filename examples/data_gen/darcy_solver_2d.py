# Author: Jonathan Siegel
"""Sine Galerkin solvers for -div(A grad u) = f on (0, 1)^2, u = 0.

NumPy inputs use the CPU by default. Pass device="cuda" to use PyTorch on
an NVIDIA GPU, or supply PyTorch tensors to keep inputs and outputs on their
device. Coefficients include boundary nodes; forcing/solution grids contain
only interior nodes. The coefficient interpolant is piecewise bilinear.
"""

from contextlib import nullcontext
from math import pi
import numpy as np


class _Backend:
    """Small NumPy/PyTorch adapter; torch remains optional for NumPy users."""

    def __init__(self, reference):
        self.torch = type(reference).__module__.startswith("torch")
        if self.torch:
            import torch
            self.xp = torch
            self.device = reference.device
        else:
            self.xp = np
        self.dtype = reference.dtype

    def zeros(self, shape, dtype=None):
        kwargs = {"device": self.device} if self.torch else {}
        return self.xp.zeros(shape, dtype=dtype or self.dtype, **kwargs)

    def arange(self, n):
        kwargs = {"device": self.device} if self.torch else {}
        return self.xp.arange(n, **kwargs)

    def flip(self, a, axis):
        return self.xp.flip(a, (axis,)) if self.torch else np.flip(a, axis)

    def cat(self, parts, axis):
        return self.xp.cat(parts, dim=axis) if self.torch else np.concatenate(parts, axis)

    def fft(self, a):
        return self.xp.fft.fft2(a, norm="forward")

    def ifft(self, a):
        return self.xp.fft.ifft2(a, norm="forward").real

    def frequencies(self, n):
        # Integer ordering avoids roundoff in the modular coefficient indices.
        k = self.arange(n)
        return self.xp.where(k <= (n - 1) // 2, k, k - n)


def _next_power_of_two(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def extend_symmetrically(a):
    """Even periodic extension of a coefficient grid including its boundary."""
    b = _Backend(a)
    out = b.cat((a, b.flip(a[1:-1, :], 0)), 0)
    return b.cat((out, b.flip(out[:, 1:-1], 1)), 1)


def extend_antisymmetrically(u):
    """Odd periodic extension of an interior grid with zero boundary values."""
    b = _Backend(u)
    n, m = u.shape
    out = b.zeros((2*n + 2, 2*m + 2))
    out[1:n+1, 1:m+1] = u
    out[n+2:, 1:m+1] = -b.flip(u, 0)
    out[1:n+1, m+2:] = -b.flip(u, 1)
    out[n+2:, m+2:] = b.flip(b.flip(u, 0), 1)
    return out


def _resize_axis(a, size, axis, extend):
    """Resize one Fourier axis, splitting/merging its even Nyquist mode."""
    old = a.shape[axis]
    if old == size:
        return a
    if (extend and size < old) or (not extend and size > old):
        raise ValueError("FFT extension must enlarge and projection must shrink the grid")
    b = _Backend(a)
    shape = list(a.shape)
    shape[axis] = size
    out = b.zeros(tuple(shape), a.dtype)

    def region(start, stop):
        s = [slice(None)] * a.ndim
        s[axis] = slice(start, stop)
        return tuple(s)

    cutoff = old if extend else size
    positive = (cutoff + 1) // 2
    negative = (cutoff - 1) // 2
    out[region(0, positive)] = a[region(0, positive)]
    if negative:
        out[region(size-negative, size)] = a[region(old-negative, old)]
    if cutoff % 2 == 0:
        k = cutoff // 2
        if extend:
            half = 0.5 * a[region(k, k+1)]
            out[region(k, k+1)] = half
            out[region(size-k, size-k+1)] = half
        else:
            out[region(k, k+1)] = a[region(k, k+1)] + a[region(old-k, old-k+1)]
    return out


def extend_fft(u_fft, n, m):
    """Pad Fourier coefficients, splitting all Nyquist rows/columns/corners."""
    return _resize_axis(_resize_axis(u_fft, n, 0, True), m, 1, True)


def project_fft(u_fft, n, m):
    """Truncate Fourier coefficients, merging the target Nyquist frequencies."""
    return _resize_axis(_resize_axis(u_fft, n, 0, False), m, 1, False)


def _endpoint_hat_sine_moments(k, intervals, b):
    """Integral of the left endpoint's linear hat against sin(pi*k*x)."""
    # 1/(pi*k) * (1 - sinc(k/intervals)); use a series to avoid
    # catastrophic cancellation for a low frequency on a fine coefficient grid.
    safe = b.xp.where(k == 0, 1, k)
    angle = pi * safe / intervals
    series = angle / (6 * intervals) * (1 - angle**2 / 20 + angle**4 / 840)
    direct = (1 - b.xp.sinc(safe / intervals)) / (pi * safe)
    result = b.xp.where(abs(angle) < 0.1, series, direct)
    return b.xp.where(k == 0, 0, result)


def _odd_coefficient_spectrum(a, k, l, x_filter, y_filter, b):
    """Exact Fourier projection of the odd-odd reflected bilinear field.

    Nonzero boundary traces create jumps in an odd reflection. They cannot be
    represented just by reflecting the interior samples: endpoint hat moments
    supply the missing boundary-edge and corner terms below.
    """
    nx, ny = a.shape[0] - 1, a.shape[1] - 1
    interior = extend_antisymmetrically(a[1:-1, 1:-1])
    transform = b.fft(interior)
    spectrum = transform[(k % (2*nx))[:, None], (l % (2*ny))[None, :]]
    spectrum = spectrum * x_filter[:, None] * y_filter[None, :]

    def edge_moments(values, frequencies, intervals, filter_values):
        periodic = b.zeros((2*intervals,))
        periodic[1:intervals] = values
        periodic[intervals+1:] = -b.flip(values, 0)
        transform = b.xp.fft.fft(periodic, norm="forward")
        return (1j * transform[frequencies % (2*intervals)]).real * filter_values

    k_real, l_real = (k.to(a.dtype), l.to(a.dtype)) if b.torch else (k, l)
    ex = _endpoint_hat_sine_moments(k_real, nx, b)
    ey = _endpoint_hat_sine_moments(l_real, ny, b)
    rx = b.xp.where(k % 2 == 0, -1, 1)
    ry = b.xp.where(l % 2 == 0, -1, 1)
    # A period-two odd-odd Fourier coefficient is minus the integral of
    # a(x,y)*sin(pi*k*x)*sin(pi*l*y) over the unit square.
    for ix, sign in ((0, 1), (-1, rx)):
        edge = edge_moments(a[ix, 1:-1], l, ny, y_filter)
        spectrum = spectrum - (ex * sign)[:, None] * edge[None, :]
    for iy, sign in ((0, 1), (-1, ry)):
        edge = edge_moments(a[1:-1, iy], k, nx, x_filter)
        spectrum = spectrum - edge[:, None] * (ey * sign)[None, :]
    corners = (a[0, 0] + a[-1, 0]*rx[:, None]
               + a[0, -1]*ry[None, :]
               + a[-1, -1]*rx[:, None]*ry[None, :])
    return spectrum - ex[:, None]*ey[None, :]*corners


def extend_coefficients(a, n, m, *, parity="even"):
    """Project the bilinear interpolant and pad for exact Galerkin products.

    Diagonal diffusion entries use even reflection in both coordinates.
    The cross entry uses odd reflection in both coordinates; this parity is
    necessary to preserve its mixed-gradient bilinear form on the square.
    """
    if parity not in ("even", "odd"):
        raise ValueError("parity must be 'even' or 'odd'")
    b = _Backend(a)
    a_ext = extend_symmetrically(a) if parity == "even" else None
    an, am = (2*a.shape[0] - 2, 2*a.shape[1] - 2)
    ln, lm = 4*n + 7, 4*m + 7
    k, l = b.frequencies(ln), b.frequencies(lm)
    # The Fourier transform of a linear nodal hat is sinc(k / grid_size)^2.
    # Broadcasting and modular gathers replace the original nested Python loop.
    kx, ly = (k.to(a.dtype), l.to(a.dtype)) if b.torch else (k, l)
    x = b.xp.sinc(kx / an)**2
    y = b.xp.sinc(ly / am)**2
    if parity == "odd":
        spectrum = _odd_coefficient_spectrum(a, k, l, x, y, b)
    else:
        spectrum = b.fft(a_ext)[(k % an)[:, None], (l % am)[None, :]]
        spectrum = spectrum * x[:, None] * y[None, :]
    spectrum = extend_fft(spectrum, _next_power_of_two(ln), _next_power_of_two(lm))
    return b.ifft(spectrum)


def _laplace_multiplier(reference):
    b = _Backend(reference)
    n, m = reference.shape
    k = b.frequencies(2*n + 2)
    l = b.frequencies(2*m + 2)
    squared = k[:, None]**2 + l[None, :]**2
    if b.torch:
        squared = squared.to(reference.dtype)
    squared[0, 0] = 1
    multiplier = 1 / (pi * b.xp.sqrt(squared))
    multiplier[0, 0] = 0
    return multiplier


def apply_inverse_square_root_laplace(u):
    """Apply (-Delta)^(-1/2) with homogeneous Dirichlet boundaries."""
    b = _Backend(u)
    n, m = u.shape
    return b.ifft(b.fft(extend_antisymmetrically(u)) * _laplace_multiplier(u))[1:n+1, 1:m+1]


class _Operator:
    """Per-solve spectral plan; all cached arrays stay on the input device."""

    def __init__(self, a11, a12, a22, reference):
        self.b = b = _Backend(reference)
        self.n, self.m = reference.shape
        self.large_n = _next_power_of_two(4*self.n + 7)
        self.large_m = _next_power_of_two(4*self.m + 7)
        expected = (self.large_n, self.large_m)
        if any(a is not None and tuple(a.shape) != expected for a in (a11, a12, a22)):
            raise ValueError("Coefficients must be extended to the operator's padded grid")
        self.a11, self.a12, self.a22 = a11, a12, a22
        self.scale = _laplace_multiplier(reference)
        # Broadcast vectors instead of storing two large meshgrids.
        k, l = b.frequencies(self.large_n), b.frequencies(self.large_m)
        if b.torch:
            k, l = k.to(reference.dtype), l.to(reference.dtype)
        self.dx = (pi * 1j) * k[:, None]
        self.dy = (pi * 1j) * l[None, :]

    def precondition(self, u):
        n, m = self.n, self.m
        return self.b.ifft(self.b.fft(extend_antisymmetrically(u)) * self.scale)[1:n+1, 1:m+1]

    def __call__(self, u):
        b, n, m = self.b, self.n, self.m
        # Fuse each half-preconditioner with the adjacent FFT. Six transforms
        # per operator application replace the original ten.
        spectrum = b.fft(extend_antisymmetrically(u)) * self.scale
        spectrum = extend_fft(spectrum, self.large_n, self.large_m)
        gx, gy = b.ifft(spectrum * self.dx), b.ifft(spectrum * self.dy)
        if self.a12 is None:
            fx, fy = self.a11 * gx, self.a22 * gy
        else:
            fx = self.a11 * gx + self.a12 * gy
            fy = self.a12 * gx + self.a22 * gy
        divergence = -(self.dx * b.fft(fx) + self.dy * b.fft(fy))
        spectrum = project_fft(divergence, 2*n + 2, 2*m + 2)
        out = b.ifft(spectrum * self.scale)
        # Orthogonal projection onto functions odd in both coordinates.
        pp = out[1:n+1, 1:m+1]
        nn = b.flip(b.flip(out[n+2:, m+2:], 0), 1)
        pn = b.flip(out[1:n+1, m+2:], 1)
        npart = b.flip(out[n+2:, 1:m+1], 0)
        return 0.25 * (pp + nn - pn - npart)


def apply_preconditioned_operator(a11, a12, a22, u):
    """Apply (-Delta)^(-1/2) [-div(A grad)] (-Delta)^(-1/2).

    Coefficients must already be prepared by extend_coefficients: use the
    default even parity for a11/a22 and parity="odd" for a12. Solvers reuse a
    plan internally; this standalone helper constructs a fresh plan.
    """
    return _Operator(a11, a12, a22, u)(u)


def _prepare_inputs(a11, a12, a22, f, device, dtype):
    arrays = (a11, a12, a22, f)
    tensors = [a for a in arrays if type(a).__module__.startswith("torch")]
    if device is not None or tensors:
        import torch
        device = torch.device(device) if device is not None else tensors[0].device
        if device.type not in ("cpu", "cuda"):
            raise ValueError("The torch FFT solver supports CPU and CUDA devices")
        if dtype is None:
            dtype = tensors[0].dtype if tensors else torch.float64
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Use torch.float32 or torch.float64 for the FFT solver")
        converted = []
        for a in arrays:
            if a is None:
                converted.append(None)
            else:
                if torch.is_complex(a) if isinstance(a, torch.Tensor) else np.iscomplexobj(a):
                    raise ValueError("Inputs must be real")
                converted.append(torch.as_tensor(a, device=device, dtype=dtype).detach())
        return converted, torch.no_grad()
    dtype = np.dtype(np.float64 if dtype is None else dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Use float32 or float64")
    if any(a is not None and np.iscomplexobj(a) for a in arrays):
        raise ValueError("Inputs must be real")
    return [None if a is None else np.asarray(a, dtype=dtype) for a in arrays], nullcontext()


def uniform_grid_solver_scalar(a, f, tol=1e-5, *, rtol=0.0, maxiter=None,
                               device=None, dtype=None, return_info=False):
    """Solve scalar Darcy; see uniform_grid_solver_tensor for options."""
    return uniform_grid_solver_tensor(a, None, a, f, tol, rtol=rtol,
                                     maxiter=maxiter, device=device, dtype=dtype,
                                     return_info=return_info)


def uniform_grid_solver_tensor(a11, a12, a22, f, tol=1e-5, *, rtol=0.0,
                               maxiter=None, device=None, dtype=None,
                               return_info=False):
    """Solve tensor Darcy with symmetrically preconditioned conjugate gradients.

    A = [[a11, a12], [a12, a22]] must be positive definite at every coefficient
    node. a12=None denotes zero cross diffusion. Coefficient arrays share a
    grid (at least 2x2); f is an interior grid (at least 1x1).

    Stops when the RMS *preconditioned* residual is at most
    max(tol, rtol * initial_residual). The true residual is recomputed before
    accepting convergence. Raises RuntimeError on breakdown or nonconvergence.
    maxiter defaults to 10 times the number of unknowns, capped at 10,000.

    NumPy inputs return a NumPy array unless device is specified. Tensor inputs
    return a detached torch tensor. Pass device="cuda" for NVIDIA GPUs and
    dtype=torch.float32 to trade precision for throughput (default float64 for
    NumPy inputs, otherwise the first tensor's dtype). No autograd is recorded.
    return_info=True returns (solution, {iterations, residual_rms, ...}).
    """
    if not np.isfinite(tol) or not np.isfinite(rtol) or tol < 0 or rtol < 0 or tol + rtol == 0:
        raise ValueError("Tolerances must be finite, nonnegative, and not both zero")
    scalar = a11 is a22
    (a11, a12, a22, f), context = _prepare_inputs(a11, a12, a22, f, device, dtype)
    with context:
        b = _Backend(f)
        if f.ndim != 2 or min(f.shape) < 1:
            raise ValueError("f must be a nonempty 2D interior grid")
        if a11.ndim != 2 or min(a11.shape) < 2 or a22.shape != a11.shape or (a12 is not None and a12.shape != a11.shape):
            raise ValueError("Coefficients must share a 2D grid of at least 2x2 nodes")
        invalid = ~b.xp.isfinite(f).all()
        for a in (a11, a12, a22):
            if a is not None:
                invalid = invalid | ~b.xp.isfinite(a).all()
        invalid = invalid | (a11 <= 0).any() | (a22 <= 0).any()
        if a12 is not None:
            invalid = invalid | (a11*a22 - a12*a12 <= 0).any()
        if bool(invalid):
            raise ValueError("Inputs must be finite and the diffusion tensor positive definite")
        n, m = f.shape
        if maxiter is None:
            maxiter = min(10*n*m, 10000)
        if not isinstance(maxiter, int) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        aa = extend_coefficients(a11, n, m)
        cc = aa if scalar else extend_coefficients(a22, n, m)
        ac = None if a12 is None else extend_coefficients(a12, n, m, parity="odd")
        operator = _Operator(aa, ac, cc, f)
        rhs = operator.precondition(f)
        u, info = _conjugate_gradient(operator, rhs, tol, rtol, maxiter)
        solution = operator.precondition(u)
        return (solution, info) if return_info else solution


def _conjugate_gradient(operator, rhs, tol, rtol, maxiter):
    """Solve a prepared preconditioned system, reusing its spectral plan."""
    b = _Backend(rhs)
    u = b.zeros(rhs.shape)
    r = rhs
    p = r
    rr = (r*r).mean()
    initial = float(rr)**0.5
    target = max(tol, rtol*initial)
    residual = initial
    iterations = 0
    while residual > target and iterations < maxiter:
        ap = operator(p)
        denominator = (p*ap).mean()
        # Transfer only small CG scalars to the host for breakdown checks.
        # FFTs, coefficient arrays and vectors remain on the selected device.
        scalars = b.xp.stack((rr, denominator))
        values = scalars.tolist()
        if not all(np.isfinite(v) for v in values) or values[1] <= 0:
            raise RuntimeError("CG breakdown: nonfinite residual or nonpositive curvature")
        alpha = rr / denominator
        u = u + alpha*p
        r = r - alpha*ap
        rr_new = (r*r).mean()
        residual = float(rr_new)**0.5
        iterations += 1
        if not np.isfinite(residual):
            raise RuntimeError("CG residual became nonfinite")
        if residual <= target:
            # A recursively updated residual can drift in float32.
            r = rhs - operator(u)
            rr_new = (r*r).mean()
            residual = float(rr_new)**0.5
            p = r
        else:
            p = r + (rr_new/rr)*p
        rr = rr_new
    if not np.isfinite(residual) or residual > target:
        raise RuntimeError(f"CG did not converge in {maxiter} iterations (preconditioned residual RMS={residual:.3e}, target={target:.3e})")
    info = dict(iterations=iterations, residual_rms=residual,
                initial_residual_rms=initial, tolerance=target, converged=True)
    return u, info


if __name__ == "__main__":
    n, m = 64, 64
    a = np.ones((3*n, 3*m))
    a[n:2*n, m:2*m] = 100
    u, info = uniform_grid_solver_scalar(a, np.ones((n, m)), return_info=True)
    print(info)
