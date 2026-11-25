# Author: Jonathan Siegel
#
# Generates random diffusion coefficients and RHS for the darcy equation and for the Calderon problem.

from functools import lru_cache

import numpy as np

def sample_gaussian_field(n, m, width=0.1, variance = 1.0):
  """Samples from a Gaussian random field on a regular grid.

  args:
    n, m: dimensions of the output
    width: the width of the random field

  returns:
    sample from the Gaussian field.
  """
  _, _, samples = sample_gaussian_gp_fft(n, m, dx=1.0 / n, dy=1.0 / m, length_scale = (width, width), variance=variance)
  return samples[0,:,:]

def _next_power_of_two(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def sample_scalar_diffusion(n, m, width_smooth=0.2, width_rough=0.1, variance = 0.5, spread=1.5):
  """Samples a random scalar diffusion field. The field jumps between two random fields depending
     upon another random field.

  args:
    n, m: dimensions of the output
    width_smooth: the width of the two random fields
    width_rough: the width of the switching field
    variance: the variance of the fields.
    spread: different in baseline in both regions.

  returns:
    sample from the Gaussian field.
  """
  a_1 = sample_gaussian_field(n, m, width=width_smooth, variance = variance)
  a_2 = sample_gaussian_field(n, m, width=width_smooth, variance = variance)
  a_choice = sample_gaussian_field(n, m, width=width_rough, variance = 1.0)
  return np.where(a_choice >= 0, np.exp(a_1 - spread), np.exp(a_2 + spread))

def sample_piecewise_constant_scalar_diffusion(n, m, width=0.1, variance = 4.0):
  """Samples a random scalar diffusion field. The field jumps between two values depending on a Gaussian field.

  args:
    n, m: dimensions of the output
    width: the width of the random field
    variance: the large and small values

  returns:
    sample from the Gaussian field.
  """
  a_choice = sample_gaussian_field(n, m, width=width, variance = 1.0)
  return np.where(a_choice >= 0, np.exp(-1.0 * variance), np.exp(variance))


# A Darcy run uses three covariance configurations: two for a and one for f.
# Bound the number of resident spectra when a notebook switches grid sizes.
@lru_cache(maxsize=3)
def _gaussian_sqrt_spectrum(nx, ny, dx, dy, ell_x, ell_y, variance, max_embedding_attempts):
    """Prepare the deterministic FFT filter. Random draws never enter the cache."""
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive.")
    if variance < 0:
        raise ValueError("variance must be nonnegative.")

    if ell_x <= 0 or ell_y <= 0:
        raise ValueError("Length scales must be positive.")

    # Embedding dimensions must be large enough that every displacement
    # within the requested grid appears without periodic wrap-around.
    mx = _next_power_of_two(max(1, 2 * (nx - 1)))
    my = _next_power_of_two(max(1, 2 * (ny - 1)))

    for _ in range(max_embedding_attempts):
        # Wrapped displacements on the embedding grid.
        ix = np.minimum(np.arange(mx), mx - np.arange(mx))
        iy = np.minimum(np.arange(my), my - np.arange(my))

        rx = dx * ix[:, None]
        ry = dy * iy[None, :]

        # First row/column of the block-circulant covariance matrix.
        covariance_lags = variance * np.exp(
            -0.5 * ((rx / ell_x) ** 2 + (ry / ell_y) ** 2)
        )

        # Eigenvalues of the circulant covariance matrix.
        eigenvalues = np.fft.fft2(covariance_lags).real

        scale = max(np.max(eigenvalues), 1.0)
        tolerance = 1e-12 * scale

        if np.min(eigenvalues) >= -tolerance:
            return np.sqrt(np.maximum(eigenvalues, 0.0))

        # A larger embedding often removes negative eigenvalues.
        mx *= 2
        my *= 2
    raise RuntimeError(
        "Could not find a positive-semidefinite circulant embedding. "
        "Try increasing max_embedding_attempts."
    )


def sample_gaussian_gp_fft(
    nx,
    ny,
    dx=1.0,
    dy=1.0,
    length_scale=(1.0, 1.0),
    variance=1.0,
    n_samples=1,
    seed=None,
    max_embedding_attempts=8,
):
    """
    Sample a zero-mean stationary Gaussian process on a 2D regular grid.

    The covariance kernel is

        k(r_x, r_y) = variance * exp(
            -0.5 * ((r_x / ell_x)**2 + (r_y / ell_y)**2)
        ).

    Circulant embedding and FFTs are used, so no dense covariance matrix
    or Cholesky factorization is required.

    Returns
    -------
    x : ndarray, shape (nx,)
        Coordinates along the first dimension.
    y : ndarray, shape (ny,)
        Coordinates along the second dimension.
    samples : ndarray, shape (n_samples, nx, ny)
        GP samples evaluated on the regular grid.
    """
    ell_x, ell_y = length_scale
    sqrt_spectrum = _gaussian_sqrt_spectrum(
        nx, ny, dx, dy, ell_x, ell_y, variance, max_embedding_attempts
    )
    mx, my = sqrt_spectrum.shape
    rng = np.random.default_rng(seed)

    # Filtering real white noise by sqrt of the covariance spectrum produces
    # real GP samples with the requested covariance.
    white_noise = rng.standard_normal((n_samples, mx, my))

    spectrum = np.fft.fft2(white_noise, axes=(-2, -1))
    spectrum *= sqrt_spectrum

    embedded_samples = np.fft.ifft2(
        spectrum,
        axes=(-2, -1),
    ).real

    # Crop the requested nonperiodic grid from the embedding.
    samples = embedded_samples[:, :nx, :ny]

    x = dx * np.arange(nx)
    y = dy * np.arange(ny)

    return x, y, samples

if __name__ == '__main__':
  n, m = (10, 10)
  u = sample_gaussian_field(n, m)
  print(u)
