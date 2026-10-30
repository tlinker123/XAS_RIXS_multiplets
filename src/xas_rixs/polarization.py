"""
Photon polarisation and scattering geometry, in the conventions of edrixs.

The scattering plane is the local *zx* plane of ``scatter_axis`` (identity by
default).  ``theta`` is measured from the sample surface (the *x* axis) and
``phi`` is the azimuth about *z*, so ``thin = thout = pi/2`` is normal
incidence / back-scattering and the scattering angle is ``2theta = thin +
thout``.  A linear polarisation is given by its angle ``alpha`` to the
scattering plane: ``alpha = 0`` is pi (LH), ``alpha = pi/2`` is sigma (LV).

A polarisation *spec* is any of

``'LH'`` / ``'pi'``, ``'LV'`` / ``'sigma'``
    linear, in / perpendicular to the scattering plane
``'left'`` / ``'C+'``, ``'right'`` / ``'C-'``
    circular, edrixs's helicity convention, ``(e_pi +/- i e_sigma)/sqrt 2``
``('linear', alpha)`` or a bare number ``alpha``
    linear at ``alpha`` radians from the scattering plane
``'isotropic'``
    the average over three orthogonal polarisations (five quadrupolar
    components), i.e. a powder or an unoriented sample
``'unpolarized'``
    pi and sigma: averaged for the incident photon, *summed* for the scattered
    one (a spectrometer without polarisation analysis)
a length-3 vector
    an explicit (complex) polarisation in the global frame
"""

from __future__ import annotations

__all__ = ['polarization_terms', 'polarization_label']

import numpy as np

import edrixs

_ALIASES = {'lh': 0.0, 'pi': 0.0, 'lv': np.pi / 2, 'sigma': np.pi / 2}
_CIRC = {'left': 'left', 'c+': 'left', 'right': 'right', 'c-': 'right'}


def _vector(theta, phi, axis, direction, kind, alpha=0.0):
    e_pi = edrixs.linear_polvec(theta, phi, 0.0, axis, direction=direction)
    e_sig = edrixs.linear_polvec(theta, phi, np.pi / 2, axis, direction=direction)
    if kind == 'linear':
        return np.cos(alpha) * e_pi + np.sin(alpha) * e_sig + 0j
    if kind == 'left':
        return (e_pi + 1j * e_sig) / np.sqrt(2.0)
    return (e_pi - 1j * e_sig) / np.sqrt(2.0)


def polarization_terms(spec, theta: float, phi: float = 0.0, scatter_axis=None,
                       direction: str = 'in', npol: int = 3) -> list:
    """Expand a polarisation spec into ``[(vector, weight), ...]``.

    The intensity is ``sum(weight * |<f| vector . T |i>|**2)``.  Vectors have
    ``npol`` components: Cartesian for a dipolar transition (``npol = 3``),
    the edrixs quadrupolar basis for ``npol = 5`` (which also needs the photon
    wave vector, taken from the same geometry).
    """
    axis = np.eye(3) if scatter_axis is None else np.asarray(scatter_axis, float)
    terms = []
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ('isotropic', 'iso', 'powder'):
            return [(np.eye(npol, dtype=complex)[k], 1.0 / npol) for k in range(npol)]
        if s in ('unpolarized', 'unpolarised', 'none'):
            w = 0.5 if direction == 'in' else 1.0
            terms = [(_vector(theta, phi, axis, direction, 'linear', a), w)
                     for a in (0.0, np.pi / 2)]
        elif s in _ALIASES:
            terms = [(_vector(theta, phi, axis, direction, 'linear', _ALIASES[s]), 1.0)]
        elif s in _CIRC:
            terms = [(_vector(theta, phi, axis, direction, _CIRC[s]), 1.0)]
        else:
            raise ValueError(f"unknown polarisation {spec!r}")
    elif np.isscalar(spec):
        terms = [(_vector(theta, phi, axis, direction, 'linear', float(spec)), 1.0)]
    elif isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
        kind = spec[0].strip().lower()
        if kind == 'linear':
            terms = [(_vector(theta, phi, axis, direction, 'linear', float(spec[1])), 1.0)]
        else:
            return polarization_terms(kind, theta, phi, axis, direction, npol)
    else:
        vec = np.asarray(spec, dtype=complex).ravel()
        if vec.size != 3:
            raise ValueError(f"a polarisation vector needs 3 components, got {spec!r}")
        terms = [(vec, 1.0)]

    if npol == 5:
        kvec = edrixs.unit_wavevector(theta, phi, axis, direction=direction)
        terms = [(edrixs.quadrupole_polvec(v, kvec), w) for v, w in terms]
    return terms


def polarization_label(spec) -> str:
    """Short text for file headers and plot legends."""
    if isinstance(spec, str):
        return spec
    if np.isscalar(spec):
        return f"linear({np.degrees(float(spec)):.0f}deg)"
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
        return f"{spec[0]}({np.degrees(float(spec[1])):.0f}deg)"
    return 'vector'
