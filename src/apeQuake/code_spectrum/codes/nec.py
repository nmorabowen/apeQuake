"""NEC-15 (Ecuador, NEC-SE-DS) elastic design spectrum.

Source: *NEC-SE-DS, Peligro sismico y requisitos de diseno sismo resistente*
(NEC-15).  Section / table / page numbers below are the printed ones.

* Tables 3, 4, 5 (3.2.2, pp. 31-32): site coefficients ``Fa``, ``Fd``, ``Fs``.
* 3.3.1 (pp. 32-35): elastic horizontal acceleration spectrum, ``eta``, ``r``,
  ``T0``, ``Tc``, ``TL``.
* 3.3.2 (p. 36): elastic displacement spectrum.
* 3.4 (p. 37): vertical component (``Ev >= 2/3 Eh``).
* 6.3.2 (p. 61): design base shear ``V = I Sa(Ta) W / (R phiP phiE)``.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..base import G, CodeSpectrumModel, register_code

#: Zone-factor columns of Tables 3-5 (zones I..VI; the last one is ">= 0.50").
Z_COLUMNS = (0.15, 0.25, 0.30, 0.35, 0.40, 0.50)

#: Table 3 (p. 31): Fa, short-period amplification.
FA_TABLE = {
    "A": (0.90, 0.90, 0.90, 0.90, 0.90, 0.90),
    "B": (1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
    "C": (1.40, 1.30, 1.25, 1.23, 1.20, 1.18),
    "D": (1.60, 1.40, 1.30, 1.25, 1.20, 1.12),
    "E": (1.80, 1.40, 1.25, 1.10, 1.00, 0.85),
}
#: Table 4 (p. 31): Fd, amplification of the displacement spectrum.
FD_TABLE = {
    "A": (0.90, 0.90, 0.90, 0.90, 0.90, 0.90),
    "B": (1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
    "C": (1.36, 1.28, 1.19, 1.15, 1.11, 1.06),
    "D": (1.62, 1.45, 1.36, 1.28, 1.19, 1.11),
    "E": (2.10, 1.75, 1.70, 1.65, 1.60, 1.50),
}
#: Table 5 (p. 32): Fs, non-linear soil behaviour.
FS_TABLE = {
    "A": (0.75, 0.75, 0.75, 0.75, 0.75, 0.75),
    "B": (0.75, 0.75, 0.75, 0.75, 0.75, 0.75),
    "C": (0.85, 0.94, 1.02, 1.06, 1.11, 1.23),
    "D": (1.02, 1.06, 1.11, 1.19, 1.28, 1.40),
    "E": (1.50, 1.60, 1.70, 1.80, 1.90, 2.00),
}

#: Spectral amplification eta by region (3.3.1, p. 34).  Esmeraldas is a coastal
#: province but takes the Sierra value; Galapagos takes 2.48 as well.
ETA_BY_REGION = {
    "costa": 1.80,
    "sierra": 2.48,
    "oriente": 2.60,
    "galapagos": 2.48,
    "esmeraldas": 2.48,
}

#: Cap of TL for site classes D and E (3.3.1 note p. 35 and 3.3.2 note p. 36) [s].
TL_MAX_DE = 4.0


def _check_site_class(site_class: str) -> str:
    sc = str(site_class).strip().upper()
    if sc == "F":
        raise ValueError(
            "Site class F has no tabulated Fa, Fd, Fs: NEC-SE-DS requires a "
            "site-specific response study (sections 3.2 and 10.5.4)"
        )
    if sc not in FA_TABLE:
        raise ValueError(f"site_class must be one of 'A'..'E', got {site_class!r}")
    return sc


def site_coefficients(z: float, site_class: str) -> tuple[float, float, float]:
    """``(Fa, Fd, Fs)`` from Tables 3-5, linear in Z, clamped outside 0.15-0.50."""
    sc = _check_site_class(site_class)
    fa, fd, fs = (
        float(np.interp(z, Z_COLUMNS, tab[sc])) for tab in (FA_TABLE, FD_TABLE, FS_TABLE)
    )
    return fa, fd, fs


@register_code("NEC", "NEC15", "NEC-SE-DS")
class NECSpectrum(CodeSpectrumModel):
    """NEC-SE-DS (NEC-15) elastic design spectrum, 5 % damping, in g.

    Parameters
    ----------
    z : float
        Seismic zone factor Z (peak rock acceleration, fraction of g, > 0).
        Table values are interpolated linearly between the columns
        0.15, 0.25, 0.30, 0.35, 0.40, >= 0.50 and clamped outside them.
    site_class : {"A", "B", "C", "D", "E"}
        Soil profile (Table 2).  ``"F"`` raises: it needs a site-specific study.
    region : {"costa", "sierra", "oriente", "galapagos", "esmeraldas"}
        Sets ``eta`` (3.3.1): Costa 1.80 (Esmeraldas excluded), Sierra,
        Esmeraldas and Galapagos 2.48, Oriente 2.60.  Not used for ``eta`` if
        ``eta`` is given (it is then only reported by :meth:`parameters`).
    eta : float, optional
        Override of the regional spectral amplification ratio.
    fa, fd, fs : float, optional
        Override the table values (e.g. from a microzonation study, 3.3.1).
    component : {"horizontal", "vertical"}
        ``"vertical"`` scales the whole horizontal spectrum by 2/3 (3.4.2: 2/3
        is the *minimum* factor of the general case; the code gives no period
        range for it).
    ascending_branch : bool
        ``True`` (default): the full Fig. 3 curve, with the left branch
        ``Z Fa (1 + (eta - 1) T / T0)`` for ``T < T0`` (so ``Sa(0) = Z Fa``,
        the zero-period / PGA-level value; the right one for comparing with a
        record).  NEC allows that branch **only for modes other than the
        fundamental**: for static analysis and the fundamental mode the
        plateau ``eta Z Fa`` extends down to T = 0 (appendix 10.1.2), which is
        ``False``.  :meth:`reduced_sa` (force-based design) always uses the
        plateau.

    Example
    -------
    >>> m = NECSpectrum(z=0.40, site_class="D", region="sierra")
    >>> round(float(m.sa(0.5)[0]), 4)   # plateau 2.48 * 0.40 * 1.20
    1.1904
    >>> round(float(m.sa(0.0)[0]), 4)    # Sa(0) = Z Fa
    0.48
    """

    code = "NEC-15"

    def __init__(
        self,
        *,
        z: float,
        site_class: str,
        region: str = "sierra",
        eta: float | None = None,
        fa: float | None = None,
        fd: float | None = None,
        fs: float | None = None,
        component: str = "horizontal",
        ascending_branch: bool = True,
    ) -> None:
        z = float(z)
        if not np.isfinite(z) or z <= 0.0:
            raise ValueError(f"z must be a finite value > 0, got {z}")
        sc = _check_site_class(site_class)

        reg = str(region).strip().lower()
        if eta is None:
            if reg not in ETA_BY_REGION:
                raise ValueError(
                    f"region must be one of {sorted(ETA_BY_REGION)}, got {region!r}"
                )
            eta = ETA_BY_REGION[reg]
        eta = float(eta)
        if not np.isfinite(eta) or eta <= 1.0:
            raise ValueError(f"eta must be a finite value > 1, got {eta}")

        t_fa, t_fd, t_fs = site_coefficients(z, sc)
        fa = t_fa if fa is None else float(fa)
        fd = t_fd if fd is None else float(fd)
        fs = t_fs if fs is None else float(fs)
        for name, val in (("fa", fa), ("fd", fd), ("fs", fs)):
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be a finite value > 0, got {val}")

        comp = str(component).strip().lower()
        if comp not in ("horizontal", "vertical"):
            raise ValueError(
                f"component must be 'horizontal' or 'vertical', got {component!r}"
            )

        self.z = z
        self.site_class = sc
        self.region = reg
        self.eta = eta
        self.fa, self.fd, self.fs = fa, fd, fs
        self.component = comp
        self.ascending_branch = bool(ascending_branch)
        self.r = 1.5 if sc == "E" else 1.0  # 3.3.1, p. 34
        self.t0 = 0.10 * fs * fd / fa  # p. 35
        self.tc = 0.55 * fs * fd / fa  # p. 34
        tl = 2.4 * fd  # p. 34
        self.tl = min(tl, TL_MAX_DE) if sc in ("D", "E") else tl  # notes p. 35, 36

    # ------------------------------------------------------------------ #
    def _horizontal(self, T: np.ndarray, ascending: bool) -> np.ndarray:
        plateau = self.eta * self.z * self.fa
        with np.errstate(divide="ignore", invalid="ignore"):
            tail = plateau * (self.tc / T) ** self.r
        sa = np.where(T <= self.tc, plateau, tail)
        if ascending:
            left = self.z * self.fa * (1.0 + (self.eta - 1.0) * T / self.t0)
            sa = np.where(T < self.t0, left, sa)
        return sa

    @property
    def _scale(self) -> float:
        return 2.0 / 3.0 if self.component == "vertical" else 1.0

    def sa(self, T: float | Iterable[float]) -> np.ndarray:
        """Elastic Sa(T) in g (see the class docstring for the T < T0 behaviour)."""
        T = self._periods(T)
        return self._scale * self._horizontal(T, self.ascending_branch)

    def sd(self, T: float | Iterable[float]) -> np.ndarray:
        """Elastic displacement spectrum [m], NEC-SE-DS 3.3.2.

        ``Sd = Sa g (T / 2 pi)^2`` up to ``TL`` and constant (``= Sd(TL)``)
        beyond it, as drawn in Fig. 4.  ``TL`` is capped at 4 s for classes D, E.
        """
        T = self._periods(T)
        Te = np.minimum(T, self.tl)
        sa = self._scale * self._horizontal(Te, self.ascending_branch)
        return sa * G * (Te / (2.0 * np.pi)) ** 2

    def reduced_sa(
        self,
        T: float | Iterable[float],
        *,
        r_factor: float,
        phi_p: float = 1.0,
        phi_e: float = 1.0,
        importance: float = 1.0,
    ) -> np.ndarray:
        """Design (inelastic) ordinate ``I Sa(T) / (R phiP phiE)`` in g.

        This is the factor that multiplies the reactive weight ``W`` in the
        force-based base shear ``V = I Sa(Ta) W / (R phiP phiE)`` (NEC-SE-DS
        6.3.2, p. 61; ``Sa`` there is the 3.3.1 spectrum, the text's reference
        to 3.3.2 is a known erratum).  It always uses the horizontal spectrum
        with the plateau extended to T = 0 (no ascending branch), so no special
        handling of ``T < T0`` is needed: static analysis and the fundamental
        mode use the plateau (appendix 10.1.2).
        """
        for name, val in (
            ("r_factor", r_factor),
            ("phi_p", phi_p),
            ("phi_e", phi_e),
            ("importance", importance),
        ):
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be a finite value > 0, got {val}")
        T = self._periods(T)
        sa = self._horizontal(T, ascending=False)
        return importance * sa / (r_factor * phi_p * phi_e)

    def parameters(self) -> dict[str, Any]:
        return {
            "Z": self.z,
            "eta": self.eta,
            "Fa": self.fa,
            "Fd": self.fd,
            "Fs": self.fs,
            "r": self.r,
            "T0": self.t0,
            "Tc": self.tc,
            "TL": self.tl,
            "site_class": self.site_class,
            "region": self.region,
        }

    def knee_periods(self) -> tuple[float, ...]:
        return (self.t0, self.tc, self.tl)
