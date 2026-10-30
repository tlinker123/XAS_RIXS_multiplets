"""
Line shapes and Krylov tools: Voigt broadening of pole lists, and the
Lanczos continued fraction used when a space is too large to diagonalise.
"""

from __future__ import annotations

__all__ = ['broaden', 'gamma_at', 'lanczos', 'continued_fraction',
           'gauss_convolve']

import numpy as np
from scipy.special import wofz

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _voigt(x: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    """Area-normalised Voigt profile (Lorentzian HWHM ``gamma``, Gaussian ``sigma``)."""
    if sigma <= 0:
        return (gamma / np.pi) / (x * x + gamma * gamma)
    z = (x + 1j * gamma) / (sigma * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def gamma_at(spec, energies):
    """Broadening at ``energies``: a constant, or interpolated breakpoints.

    ``spec`` is either a number or a sequence of ``[energy, gamma]`` pairs.
    """
    energies = np.asarray(energies, dtype=float)
    if np.isscalar(spec) or np.ndim(spec) == 0:
        return np.full(energies.shape, float(spec))
    arr = np.atleast_2d(np.asarray(spec, dtype=float))
    if arr.shape[1] != 2:
        raise ValueError("broadening breakpoints must be [[energy, gamma], ...]")
    order = np.argsort(arr[:, 0])
    return np.interp(energies, arr[order, 0], arr[order, 1])


def broaden(pole_e: np.ndarray, pole_w: np.ndarray, grid: np.ndarray,
            gamma, gauss_fwhm: float = 0.0, chunk: int = 4096) -> np.ndarray:
    """Sum of Voigt profiles, one per pole.

    ``gamma`` is a scalar Lorentzian HWHM or one value per pole.
    """
    sigma = gauss_fwhm * _FWHM_TO_SIGMA
    per_pole = np.ndim(gamma) > 0
    gam = np.broadcast_to(np.asarray(gamma, dtype=float).ravel(),
                          (len(pole_e),)) if per_pole else float(gamma)
    out = np.zeros_like(grid, dtype=float)
    for start in range(0, len(pole_e), chunk):
        sl = slice(start, start + chunk)
        e, w = pole_e[sl], pole_w[sl]
        g = gam[sl][None, :] if per_pole else gam
        out += (_voigt(grid[:, None] - e[None, :], sigma, g) * w[None, :]).sum(axis=1)
    return out


def gauss_convolve(grid: np.ndarray, y: np.ndarray, fwhm: float) -> np.ndarray:
    """Convolve a spectrum sampled on a uniform grid with a Gaussian."""
    if fwhm <= 0:
        return y
    dx = grid[1] - grid[0]
    sigma = fwhm * _FWHM_TO_SIGMA
    n = max(int(np.ceil(4 * sigma / dx)), 1)
    t = np.arange(-n, n + 1) * dx
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.full(n, y[0]), y, np.full(n, y[-1])])
    return np.convolve(pad, k, mode='same')[n:-n]


def lanczos(matvec, v0: np.ndarray, nkryl: int, tol: float = 1e-8,
             reorth: bool = True):
    """Hermitian Lanczos tridiagonalisation of ``matvec`` starting from ``v0``.

    With ``reorth`` the Krylov basis is kept fully orthogonal, which costs one
    pass over all stored vectors per step; without it only the previous two
    vectors are stored.
    """
    n = v0.size
    m = int(min(nkryl, n))
    Q = np.zeros((m if reorth else 2, n), dtype=complex)
    alpha = np.zeros(m)
    beta = np.zeros(m)
    Q[0] = v0 / np.linalg.norm(v0)
    used = m
    for k in range(m):
        cur, prev = (k, k - 1) if reorth else (k % 2, (k - 1) % 2)
        w = matvec(Q[cur])
        alpha[k] = np.real(np.vdot(Q[cur], w))
        w = w - alpha[k] * Q[cur]
        if k > 0:
            w -= beta[k] * Q[prev]
        if reorth:
            c = Q[:k + 1].conj() @ w
            w -= c @ Q[:k + 1]
        b = np.linalg.norm(w)
        if k + 1 < m:
            if b < tol:
                used = k + 1
                break
            beta[k + 1] = b
            Q[(k + 1) if reorth else (k + 1) % 2] = w / b
    return alpha[:used], beta[:used]


def continued_fraction(alpha: np.ndarray, beta: np.ndarray, z: np.ndarray) -> np.ndarray:
    """``<v|(z - H)^-1|v>`` for a normalised ``v`` from Lanczos coefficients."""
    g = np.zeros_like(z, dtype=complex)
    for k in range(len(alpha) - 1, -1, -1):
        if k == len(alpha) - 1:
            g = 1.0 / (z - alpha[k])
        else:
            g = 1.0 / (z - alpha[k] - beta[k + 1] ** 2 * g)
    return g
