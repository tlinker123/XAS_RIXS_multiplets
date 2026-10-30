"""
XAS and RIXS spectra of a solved ion.

XAS (Fermi golden rule, dipolar or quadrupolar)

.. math::

    \\mu(\\omega) = \\sum_i p_i \\sum_n |\\langle n|\\epsilon\\cdot T|i\\rangle|^2
                  \\, L_{\\Gamma_c}(\\omega - E_n + E_i)

RIXS (Kramers-Heisenberg)

.. math::

    I(\\omega_{in}, \\omega_{loss}) = \\sum_i p_i \\sum_f \\Big|\\sum_n
        \\frac{\\langle f|(\\epsilon'\\cdot T)^\\dagger|n\\rangle
              \\langle n|\\epsilon\\cdot T|i\\rangle}
             {\\omega_{in} - E_n + E_i + i\\Gamma_c}\\Big|^2
        L_{\\Gamma_f}(\\omega_{loss} - E_f + E_i)

``T`` are the edrixs transition operators, so absolute intensities are on
the edrixs scale and can be compared between polarisations, geometries and
edges of one ion.
"""

from __future__ import annotations

__all__ = ['xas', 'rixs', 'XASResult', 'RIXSResult', 'save_result',
           'plot_xas', 'plot_rixs_map']

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from .lineshape import broaden, gamma_at
from .polarization import polarization_label, polarization_terms
from .solver import IonSolution


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    if isinstance(obj, (list, tuple)):
        return [_jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


# ----------------------------------------------------------------------------
# XAS
# ----------------------------------------------------------------------------

@dataclass
class XASResult:
    """Output of :func:`xas`."""
    omega: np.ndarray            #: photon energy (eV)
    intensity: np.ndarray        #: broadened absorption
    poles: np.ndarray            #: (n, 2) array of (photon energy, weight)
    ion: IonSolution
    info: dict

    @property
    def label(self) -> str:
        return self.info['label']

    def save(self, prefix=None, outdir='.'):
        return save_result(self, prefix=prefix, outdir=outdir)


def xas(ion: IonSolution, omega=None, *, pol='isotropic', thin: float = np.pi / 2,
        phi: float = 0.0, scatter_axis=None, gamma_c=None, gauss_fwhm: float = 0.0,
        n_points: int = 2000, normalize: bool = False) -> XASResult:
    """Absorption spectrum of ``ion``.

    Parameters
    ----------
    omega : array, optional
        Photon energies on the axis of ``ion`` (see ``IonModel.reference``).
        Default: every pole, padded by 5 eV.
    pol : polarisation spec
        See :mod:`xas_rixs.polarization`; ``'isotropic'`` by default.  XMCD is
        ``xas(pol='left') - xas(pol='right')`` with a field in ``ext_B``.
    thin, phi, scatter_axis : geometry
        Incidence angle from the sample surface, azimuth, and the frame of the
        scattering plane.  Only used by a polarisation that is not isotropic
        (and by a quadrupolar edge, whose operator depends on the wave vector).
    gamma_c : float or [[energy, gamma], ...], optional
        Lorentzian HWHM; breakpoints give an energy-dependent width on the
        photon-energy axis (the L2 edge is broader than L3, say).  Default:
        ``ion.gamma_c``.
    gauss_fwhm : float
        Gaussian FWHM of the instrumental resolution.
    """
    gamma_spec = ion.gamma_c if gamma_c is None else gamma_c
    terms = polarization_terms(pol, thin, phi, scatter_axis, 'in', ion.npol)
    pe, pw = [], []
    for g, pg in zip(ion.gs, ion.p):
        for vec, wt in terms:
            theta, c, _ = ion.absorption_poles(g, vec)
            pe.append(theta - ion.E_i[g])
            pw.append(pg * wt * np.abs(c) ** 2)
    pe, pw = np.concatenate(pe), np.concatenate(pw)
    keep = pw > 1e-12 * max(pw.max(), 1e-300)
    pe, pw = ion.from_raw(pe[keep]), pw[keep]
    order = np.argsort(pe)
    poles = np.column_stack([pe[order], pw[order]])

    if omega is None:
        omega = np.linspace(poles[0, 0] - 5.0, poles[-1, 0] + 5.0, n_points)
    omega = np.asarray(omega, dtype=float)
    gam = gamma_at(gamma_spec, poles[:, 0]) if np.ndim(gamma_spec) else float(gamma_spec)
    intensity = broaden(poles[:, 0], poles[:, 1], omega, gam, gauss_fwhm)
    info = dict(label=f"{ion.info['label']}_XAS_{polarization_label(pol)}",
                pol=polarization_label(pol), thin=thin, phi=phi,
                gamma_c=_jsonable(gamma_spec), gauss_fwhm=gauss_fwhm,
                sum_weights=float(poles[:, 1].sum()), norm=1.0)
    if normalize and intensity.max() > 0:
        info['norm'] = float(intensity.max())
        intensity = intensity / intensity.max()
    return XASResult(omega=omega, intensity=intensity, poles=poles, ion=ion, info=info)


# ----------------------------------------------------------------------------
# RIXS
# ----------------------------------------------------------------------------

@dataclass
class RIXSResult:
    """Output of :func:`rixs`."""
    ominc: np.ndarray            #: incident photon energies (eV), ion's axis
    eloss: np.ndarray            #: energy-loss axis (eV)
    intensity: np.ndarray        #: (len(ominc), len(eloss)) broadened map
    pole_e: np.ndarray           #: energy loss of every (final, initial) pair
    pole_w: np.ndarray           #: (len(ominc), len(pole_e)) weights
    ion: IonSolution
    info: dict

    @property
    def label(self) -> str:
        return self.info['label']

    def spectrum(self, k: int = 0) -> np.ndarray:
        """The loss spectrum at the ``k``-th incident energy."""
        return self.intensity[k]

    def save(self, prefix=None, outdir='.'):
        return save_result(self, prefix=prefix, outdir=outdir)


def rixs(ion: IonSolution, ominc, eloss=None, *, pol_in='LH', pol_out='unpolarized',
         thin: float = np.radians(75), thout: float = np.radians(75), phi: float = 0.0,
         scatter_axis=None, gamma_c=None, gamma_f: float = 0.05,
         gauss_fwhm: float = 0.0, skip_elastic: bool = False,
         n_points: int = 1500, normalize: bool = False) -> RIXSResult:
    """RIXS of ``ion`` at the incident energies ``ominc``.

    Parameters
    ----------
    ominc : float or array
        Incident photon energies on the axis of ``ion`` -- the same axis as
        :func:`xas`, so a peak of the XAS can be picked off directly.
    eloss : array, optional
        Energy-loss grid.  Default: from -0.5 eV to past the last final
        state with weight.
    pol_in, pol_out : polarisation specs
        See :mod:`xas_rixs.polarization`.  The default -- pi incident, no
        analysis of the scattered beam -- is the usual experiment.
    thin, thout, phi, scatter_axis : geometry
        Angles from the sample surface; the scattering angle is
        ``thin + thout`` (150 degrees by default).
    gamma_c : float or [[energy, gamma], ...], optional
        Intermediate-state Lorentzian HWHM, constant or as breakpoints on the
        incident-energy axis.  Default ``ion.gamma_c``.
    gamma_f : float
        Final-state Lorentzian HWHM.
    gauss_fwhm : float
        Gaussian FWHM of the energy-loss resolution.
    skip_elastic : bool
        Leave out transitions back into the thermally populated states.
    """
    ominc = np.atleast_1d(np.asarray(ominc, dtype=float))
    gamma_spec = ion.gamma_c if gamma_c is None else gamma_c
    gam_c = gamma_at(gamma_spec, ominc)
    z = ion.to_raw(ominc) + 1j * gam_c                     # (n_in,)
    t_in = polarization_terms(pol_in, thin, phi, scatter_axis, 'in', ion.npol)
    t_out = polarization_terms(pol_out, thout, phi, scatter_axis, 'out', ion.npol)
    keep_basis = ion.method != 'exact'

    E_i = ion.E_i
    n_f = len(E_i)
    pole_e = np.concatenate([E_i - E_i[g] for g in ion.gs])
    pole_w = np.zeros((len(ominc), len(pole_e)))
    for k, (g, pg) in enumerate(zip(ion.gs, ion.p)):
        sl = slice(k * n_f, (k + 1) * n_f)
        for vin, win in t_in:
            theta, c, R = ion.absorption_poles(g, vin, keep_basis=keep_basis)
            if len(theta) == 0:
                continue
            # amplitude of each intermediate state, per incident energy
            X = c[:, None] / (z[None, :] + E_i[g] - theta[:, None])   # (m, n_in)
            for vout, wout in t_out:
                F = ion.emission_matrix(vout, R) @ X                  # (n_f, n_in)
                pole_w[:, sl] += pg * win * wout * (np.abs(F) ** 2).T
    if skip_elastic:
        for k in range(len(ion.gs)):
            pole_w[:, k * n_f + ion.gs] = 0.0

    if eloss is None:
        strong = pole_w.max(axis=0) > 1e-4 * max(pole_w.max(), 1e-300)
        hi = pole_e[strong].max() if strong.any() else 1.0
        eloss = np.linspace(-0.5, hi + max(1.0, 10 * gamma_f), n_points)
    eloss = np.asarray(eloss, dtype=float)
    intensity = np.array([broaden(pole_e, w, eloss, gamma_f, gauss_fwhm)
                          for w in pole_w])
    info = dict(label=f"{ion.info['label']}_RIXS_{polarization_label(pol_in)}"
                      f"_{polarization_label(pol_out)}",
                pol_in=polarization_label(pol_in), pol_out=polarization_label(pol_out),
                thin=thin, thout=thout, phi=phi, gamma_c=_jsonable(gamma_spec),
                gamma_f=gamma_f, gauss_fwhm=gauss_fwhm, skip_elastic=skip_elastic,
                norm=1.0)
    if normalize and intensity.max() > 0:
        info['norm'] = float(intensity.max())
        intensity = intensity / intensity.max()
    return RIXSResult(ominc=ominc, eloss=eloss, intensity=intensity, pole_e=pole_e,
                      pole_w=pole_w, ion=ion, info=info)


# ----------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------

def save_result(res, prefix: Optional[str] = None, outdir: str = '.') -> dict:
    """Write a result to text files plus an ``_info.json`` with its provenance.

    XAS: ``<prefix>_xas.dat`` (energy, intensity) and ``_poles.dat``.
    RIXS: ``<prefix>_rixs.dat`` -- a matrix whose first row is the energy-loss
    axis and first column the incident energies -- and ``_poles.dat``.
    """
    prefix = prefix or res.label
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, prefix)
    model = res.ion.model
    written = {}
    if isinstance(res, XASResult):
        hdr = (f"{model.describe()}\npol={res.info['pol']}  "
               f"gamma_c={res.info['gamma_c']} eV (HWHM)  "
               f"gauss_fwhm={res.info['gauss_fwhm']} eV\nphoton energy (eV)   intensity")
        np.savetxt(base + '_xas.dat', np.column_stack([res.omega, res.intensity]),
                   header=hdr)
        np.savetxt(base + '_poles.dat', res.poles, header='photon energy (eV)  weight')
        written.update(spectrum=base + '_xas.dat', poles=base + '_poles.dat')
    else:
        hdr = (f"{model.describe()}\npol_in={res.info['pol_in']}  "
               f"pol_out={res.info['pol_out']}  gamma_c={res.info['gamma_c']}  "
               f"gamma_f={res.info['gamma_f']}  gauss_fwhm={res.info['gauss_fwhm']}\n"
               "row 0: energy loss (eV); column 0: incident energy (eV)")
        mat = np.zeros((len(res.ominc) + 1, len(res.eloss) + 1))
        mat[0, 1:] = res.eloss
        mat[1:, 0] = res.ominc
        mat[1:, 1:] = res.intensity
        np.savetxt(base + '_rixs.dat', mat, header=hdr)
        keep = res.pole_w.max(axis=0) > 1e-10 * max(res.pole_w.max(), 1e-300)
        np.savetxt(base + '_poles.dat',
                   np.column_stack([res.pole_e[keep], res.pole_w[:, keep].T]),
                   header='energy loss (eV)  weight at each incident energy: '
                          + ' '.join(f'{w:.4f}' for w in res.ominc))
        written.update(spectrum=base + '_rixs.dat', poles=base + '_poles.dat')

    ion_info = {k: v for k, v in res.ion.info.items()}
    payload = dict(model=_jsonable(asdict(model)), ion=_jsonable(ion_info),
                   spectrum=_jsonable(res.info))
    with open(base + '_info.json', 'w') as fh:
        json.dump(payload, fh, indent=2, default=str)
    written['info'] = base + '_info.json'
    return written


def plot_xas(results, ax=None, offset: float = 0.0, poles: bool = False):
    """Quick look at one or several :class:`XASResult` objects."""
    import matplotlib.pyplot as plt
    if isinstance(results, XASResult):
        results = [results]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5 + 0.6 * (len(results) - 1)))
    for n, res in enumerate(results):
        line, = ax.plot(res.omega, res.intensity + n * offset, label=res.label)
        if poles:
            ax.vlines(res.poles[:, 0], n * offset,
                      n * offset + res.poles[:, 1] / res.poles[:, 1].max()
                      * res.intensity.max(), color=line.get_color(), lw=0.6, alpha=0.5)
    ax.set_xlabel('photon energy (eV)')
    ax.set_ylabel('absorption (arb. units)')
    ax.legend(fontsize=8)
    ax.figure.tight_layout()
    return ax


def plot_rixs_map(res: RIXSResult, ax=None, log: bool = False):
    """Incident energy vs. energy loss map of a :class:`RIXSResult`."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    data = res.intensity
    norm = LogNorm(vmin=max(data.max() * 1e-4, 1e-300), vmax=data.max()) if log else None
    mesh = ax.pcolormesh(res.eloss, res.ominc, data, shading='auto', cmap='viridis',
                         norm=norm)
    ax.figure.colorbar(mesh, ax=ax, label='intensity (arb. units)')
    ax.set_xlabel('energy loss (eV)')
    ax.set_ylabel('incident energy (eV)')
    ax.set_title(res.label, fontsize=9)
    ax.figure.tight_layout()
    return ax
