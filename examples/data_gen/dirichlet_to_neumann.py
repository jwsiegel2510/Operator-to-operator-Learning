"""Single-side Dirichlet-to-Neumann matrices for the unit square.

Boundary basis functions are sin(k*pi*t), k=1,...,n, where t increases in the
physical coordinate along the selected side. All other sides have zero data.
The auxiliary lifts are sin(k*pi*t)*L(s), with L=1-s on bottom/left and L=s on
top/right. The map represents the OUTWARD conormal derivative nu . A grad(u),
not the Darcy velocity -A grad(u).
"""

from math import ceil, pi
import numpy as np

from .darcy_solver_2d import (
    _Backend, _Operator, _conjugate_gradient, _prepare_inputs,
    extend_coefficients,
)


def _array_like(values, reference):
    b = _Backend(reference)
    if b.torch:
        return b.xp.as_tensor(values, device=b.device, dtype=b.dtype)
    return np.asarray(values, dtype=b.dtype)


def _hat_moments(nodes, frequency, reference, powers, slope, intercept, order):
    """Integrals of nodal hats times sin/cos and powers of the linear lift.

    Gauss integration is performed separately on each coefficient cell. Only
    one-dimensional moments are needed: no large 2D quadrature grid is built.
    Frequency chunks bound temporary storage independently of the mode count.
    """
    b = _Backend(reference)
    cells = nodes - 1
    if order is None:
        # Resolve the fastest phase on each cell, with margin for float64.
        order = max(8, ceil(pi * frequency / cells) + 4)
    z, weights = np.polynomial.legendre.leggauss(order)
    z = _array_like((z + 1)/2, reference)
    weights = _array_like(weights/(2*cells), reference)
    indices = b.arange(cells)
    x = (indices[:, None] + z[None, :])/cells
    linear = intercept + slope*x
    cos = {power: b.zeros((frequency+1, nodes)) for power in powers}
    sin = {power: b.zeros((frequency+1, nodes)) for power in powers}
    chunk = max(1, min(64, 262144 // (cells*order)))
    for start in range(0, frequency+1, chunk):
        stop = min(start+chunk, frequency+1)
        k = _array_like(np.arange(start, stop), reference)
        phase = pi*k[:, None, None]*x[None, :, :]
        cosine, sine = b.xp.cos(phase), b.xp.sin(phase)
        for power in powers:
            w = weights[None, :]*linear**power
            left, right = w*(1-z)[None, :], w*z[None, :]
            for output, trig in ((cos[power], cosine), (sin[power], sine)):
                output[start:stop, :-1] += (trig*left[None, :, :]).sum(axis=2)
                output[start:stop, 1:] += (trig*right[None, :, :]).sum(axis=2)
    return cos, sin


def _trig_products(cos, sin, p, k, b):
    """Hat integrals of cos(p)cos(k), cos(p)sin(k), sin(p)cos(k), sin(p)sin(k)."""
    difference = p-k
    absolute = b.xp.abs(difference)
    cc = 0.5*(cos[absolute] + cos[p+k])
    ss = 0.5*(cos[absolute] - cos[p+k])
    signed = b.xp.sign(difference)[:, None]*sin[absolute]
    cs = 0.5*(sin[p+k] - signed)
    sc = 0.5*(sin[p+k] + signed)
    return cc, cs, sc, ss


def dirichlet_to_neumann_scalar(a, n, *, side="bottom", interior_shape=None,
                                tol=1e-10, rtol=0.0, maxiter=None, device=None,
                                dtype=None, quadrature_order=None,
                                return_info=False):
    """Return the n-by-n single-side DtN matrix for scalar diffusion a.

    See dirichlet_to_neumann_tensor for basis, normalization, and options.
    """
    return dirichlet_to_neumann_tensor(
        a, None, a, n, side=side, interior_shape=interior_shape, tol=tol,
        rtol=rtol, maxiter=maxiter, device=device, dtype=dtype,
        quadrature_order=quadrature_order, return_info=return_info,
    )


def dirichlet_to_neumann_tensor(a11, a12, a22, n, *, side="bottom",
                                interior_shape=None, tol=1e-10, rtol=0.0,
                                maxiter=None, device=None, dtype=None,
                                quadrature_order=None, return_info=False):
    """Return a single-side Galerkin Dirichlet-to-Neumann coefficient matrix.

    On (0,1)^2 solve -div(A grad u)=0. A is symmetric positive definite and
    piecewise bilinear on the supplied coefficient grid, including boundary
    nodes. a12=None means zero cross diffusion. Array axes are physical x,y.

    The boundary datum g(t)=sum_k g_k*sin(k*pi*t) is supported on `side`, one of
    bottom (y=0), top (y=1), left (x=0), right (x=1). The returned matrix D maps
    g_k to coefficients of the weak outward conormal flux on that SAME side.
    Fluxes on the other sides are not returned. The boundary basis vanishes
    at the corners.

    The interior space spans sin(p*pi*x)*sin(q*pi*y). `interior_shape=(nx,ny)`
    specifies its mode counts (default (n,n)); it can be refined independently
    of the boundary mode count. E_k=sin(k*pi*t)*L(s) are auxiliary lifts, where
    L is linear, one on the chosen side, zero on the opposite side.

    With K the interior stiffness, B_ik=integral grad(v_i).A.grad(E_k), and
    C_jk=integral grad(E_j).A.grad(E_k), D=2*(C-B.T @ inv(K) @ B). The factor 2
    divides by the boundary sine mass 1/2. The existing FFT Galerkin operator
    and its Laplace-preconditioned CG solve each interior correction; no
    interior stiffness matrix or inverse is assembled. Coefficient preparation
    and the spectral plan are shared across columns. Couplings require storage
    O(n*nx*ny), so choose the boundary resolution accordingly.

    Lift integrals use separable composite Gauss-Legendre moments on coefficient
    cells. quadrature_order=None chooses an order from the largest frequency
    per cell; a supplied order >=2 is used on each cell (the caller must check
    quadrature convergence). This quadrature error is separate from the CG
    tolerance and the interior Galerkin error. tol and rtol control the RMS
    preconditioned residual as in uniform_grid_solver_tensor.

    NumPy inputs use NumPy unless device is specified; PyTorch tensors retain
    their backend/device. device='cuda' selects NVIDIA GPUs. dtype and maxiter
    follow the spectral solver. For float32, use a tolerance such as tol=1e-5
    rather than the float64 default. No autograd is recorded. return_info=True
    returns (D, {'side': ..., 'interior_shape': ..., 'solves': [CG info, ...]}).
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer boundary mode count")
    if side not in ("bottom", "top", "left", "right"):
        raise ValueError("side must be bottom, top, left, or right")
    if interior_shape is None:
        interior_shape = (n, n)
    if (not isinstance(interior_shape, (tuple, list)) or len(interior_shape) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in interior_shape)):
        raise ValueError("interior_shape must contain two positive integer mode counts")
    if (quadrature_order is not None and
            (not isinstance(quadrature_order, int) or isinstance(quadrature_order, bool) or quadrature_order < 2)):
        raise ValueError("quadrature_order must be an integer >= 2")
    if not np.isfinite(tol) or not np.isfinite(rtol) or tol < 0 or rtol < 0 or tol+rtol == 0:
        raise ValueError("Tolerances must be finite, nonnegative, and not both zero")
    if maxiter is None:
        maxiter = min(10*interior_shape[0]*interior_shape[1], 10000)
    if not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")

    scalar = a11 is a22
    arrays, context = _prepare_inputs(a11, a12, a22, np.zeros(interior_shape), device, dtype)
    a11, a12, a22, reference = arrays
    with context:
        b = _Backend(reference)
        if (a11.ndim != 2 or min(a11.shape) < 2 or a22.shape != a11.shape
                or (a12 is not None and a12.shape != a11.shape)):
            raise ValueError("Coefficients must share a 2D grid of at least 2x2 nodes")
        invalid = (a11 <= 0).any() | (a22 <= 0).any()
        for a in (a11, a12, a22):
            if a is not None:
                invalid = invalid | ~b.xp.isfinite(a).all()
        if a12 is not None:
            invalid = invalid | (a11*a22-a12*a12 <= 0).any()
        if bool(invalid):
            raise ValueError("Coefficients must be finite and the diffusion tensor positive definite")

        # Orient all computations with tangential coordinate first, normal
        # coordinate second. Tensor components must rotate with the axes.
        if side in ("left", "right"):
            a11, a22 = a22.T, a11.T
            a12 = None if a12 is None else a12.T
            reference = reference.T
        nt, ns = reference.shape
        slope, intercept = (-1, 1) if side in ("bottom", "left") else (1, 0)
        ct, st = _hat_moments(a11.shape[0], max(nt+n, 2*n), reference,
                             (0,), 0, 1, quadrature_order)
        cn, sn = _hat_moments(a11.shape[1], ns, reference,
                             (0, 1, 2), slope, intercept, quadrature_order)
        ct, st = ct[0], st[0]
        p, q = b.arange(nt)+1, b.arange(ns)+1
        modes = b.arange(n)+1
        # Normal integrals are contracted once for all boundary columns.
        tt = a11 @ sn[1][1:].T
        nn = a22 @ cn[0][1:].T
        tn_s = None if a12 is None else a12 @ sn[0][1:].T
        tn_c = None if a12 is None else a12 @ cn[1][1:].T
        weight_tt = a11 @ cn[2][0]
        weight_nn = a22 @ cn[0][0]
        weight_tn = None if a12 is None else a12 @ cn[1][0]
        t = _array_like(np.arange(1, nt+1)/(nt+1), reference)
        s = _array_like(np.arange(1, ns+1)/(ns+1), reference)
        tangent_sines = b.xp.sin(pi*t[:, None]*p[None, :])
        normal_sines = b.xp.sin(pi*s[:, None]*q[None, :])
        coupling = b.zeros((n, nt, ns))
        c = b.zeros((n, n))
        for column in range(n):
            k = column+1
            cc, cs, sc, ss = _trig_products(ct, st, p, k, b)
            modal = (pi*p)[:, None]*(pi*k)*(cc @ tt)
            modal = modal + slope*(ss @ nn)*(pi*q)[None, :]
            if a12 is not None:
                modal = modal + slope*(pi*p)[:, None]*(cs @ tn_s)
                modal = modal + (pi*k)*(sc @ tn_c)*(pi*q)[None, :]
            # Each interior sine-sine basis vector has mass 1/4. Store the
            # grid Riesz representative so existing spectral CG can be used.
            coupling[column] = 4*(tangent_sines @ modal @ normal_sines.T)
            cc, cs, sc, ss = _trig_products(ct, st, modes, k, b)
            c[:, column] = (pi*modes)*(pi*k)*(cc @ weight_tt) + slope**2*(ss @ weight_nn)
            if a12 is not None:
                c[:, column] += slope*((pi*modes)*(cs @ weight_tn) + (pi*k)*(sc @ weight_tn))

        aa = extend_coefficients(a11, nt, ns)
        dd = aa if scalar else extend_coefficients(a22, nt, ns)
        cross = None if a12 is None else extend_coefficients(a12, nt, ns, parity="odd")
        operator = _Operator(aa, cross, dd, reference)
        matrix = b.zeros((n, n))
        solves = []
        flat = coupling.reshape(n, nt*ns)
        mass_scale = (nt+1)*(ns+1)
        for column in range(n):
            rhs = operator.precondition(coupling[column])
            correction, info = _conjugate_gradient(operator, rhs, tol, rtol, maxiter)
            correction = operator.precondition(correction)
            matrix[:, column] = 2*(c[:, column] - flat @ correction.reshape(nt*ns)/mass_scale)
            solves.append(info)
        info = dict(side=side, interior_shape=tuple(interior_shape), solves=solves)
        return (matrix, info) if return_info else matrix
