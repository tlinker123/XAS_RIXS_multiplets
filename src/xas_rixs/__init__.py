"""
xas_rixs
========

X-ray absorption (XAS) and resonant inelastic x-ray scattering (RIXS) of
transition-metal, lanthanide and actinide ions in the single-ion
(crystal-field multiplet) model, built on
`edrixs <https://github.com/NSLS-II/edrixs>`_.

>>> from xas_rixs import IonModel, solve_ion, xas, rixs
>>> ion = solve_ion(IonModel(atom='Ni', valence=2, edge='L23', ten_dq=1.1))
>>> spec = xas(ion, pol='isotropic')                       # doctest: +SKIP
>>> rmap = rixs(ion, ominc=[0.0], pol_in='LH')              # doctest: +SKIP

The ion -- Coulomb multiplets, spin-orbit coupling, crystal field, exchange
field -- is diagonalised once by :func:`solve_ion`; :func:`xas` and
:func:`rixs` then give spectra for any polarisation, geometry and broadening.
Slater integrals and spin-orbit couplings come from ``atom_data/`` (Crispy's
Cowan Hartree-Fock tabulation, 50 elements) with the edrixs atom data as a
second source.

The Anderson impurity (charge-transfer) model is in development; see the
README.
"""

from .atomic import *          # noqa: F401,F403
from .atomic import __all__ as _atomic_all
from .fock import *            # noqa: F401,F403
from .fock import __all__ as _fock_all
from .lineshape import broaden, gamma_at
from .model import IonModel, build_hamiltonians, replace_model, transition_operators
from .polarization import polarization_terms
from .solver import IonSolution, solve_ion
from .spectra import (RIXSResult, XASResult, plot_rixs_map, plot_xas, rixs,
                      save_result, xas)

__version__ = '0.1.0'

__all__ = ['IonModel', 'IonSolution', 'solve_ion', 'xas', 'rixs', 'XASResult',
           'RIXSResult', 'save_result', 'plot_xas', 'plot_rixs_map',
           'build_hamiltonians', 'replace_model', 'transition_operators',
           'polarization_terms', 'broaden', 'gamma_at'] + _atomic_all + _fock_all
