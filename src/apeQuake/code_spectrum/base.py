"""Base classes and registry for building-code (design) response spectra.

Every code edition is one :class:`CodeSpectrumModel` subclass that lives in
``apeQuake/code_spectrum/codes/<module>.py`` and registers itself with
:func:`register_code`.  The contract is deliberately small:

* ``sa(T)`` returns the **elastic 5 %-damped pseudo-acceleration spectrum in g**
  for periods ``T`` in seconds (scalar or array, ``T >= 0``; ``T = 0`` is the
  zero-period / PGA-level value);
* ``parameters()`` returns the scalar quantities that define the spectrum
  (``SDS``, ``Z``, ``Fa`` ... ) so they can be tabulated and cited;
* ``knee_periods()`` returns the characteristic periods (``T0``, ``Ts``,
  ``TL`` ...) so a sampling grid can include the corners exactly.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable

import numpy as np
import pandas as pd

from ..core.units import G  # standard gravity, turns Sa [g] into Sd [m]; re-exported

__all__ = [
    "G",
    "CodeSpectrumModel",
    "AsceTwoPeriodSpectrum",
    "DerivedSpectrum",
    "register_code",
    "get_model_class",
    "available_codes",
]


_REGISTRY: dict[str, type["CodeSpectrumModel"]] = {}


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def register_code(*aliases: str):
    """Class decorator: register a model under ``cls.code`` and any *aliases*.

    Lookup (:func:`get_model_class`) ignores case, spaces, dashes and dots, so
    ``"ASCE 7-16"``, ``"asce7_16"`` and ``"ASCE7-16"`` are the same key.
    """

    def deco(cls: type["CodeSpectrumModel"]) -> type["CodeSpectrumModel"]:
        for name in (cls.code, *aliases):
            _REGISTRY[_key(name)] = cls
        return cls

    return deco


def get_model_class(code: str) -> type["CodeSpectrumModel"]:
    """Model class registered for *code* (case / punctuation insensitive)."""
    try:
        return _REGISTRY[_key(code)]
    except KeyError:
        raise KeyError(
            f"Unknown code {code!r}. Available: {', '.join(available_codes())}"
        ) from None


def available_codes() -> list[str]:
    """Canonical code names of every registered model, sorted."""
    return sorted({cls.code for cls in _REGISTRY.values()})


class CodeSpectrumModel(ABC):
    """One building-code elastic design spectrum (5 % damping, units of g)."""

    #: Canonical name, e.g. ``"NEC-15"``, ``"ASCE7-16"``.
    code: ClassVar[str]

    @abstractmethod
    def sa(self, T: float | Iterable[float]) -> np.ndarray:
        """Elastic pseudo-acceleration Sa(T) in g; vectorised, ``T >= 0``."""

    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Scalar parameters that define this spectrum (floats or short strings)."""

    def knee_periods(self) -> tuple[float, ...]:
        """Corner periods [s] of the spectrum (``T0``, ``Ts``, ``TL`` ...)."""
        return ()

    # ------------------------------------------------------------------ #
    # Derived helpers shared by every code
    # ------------------------------------------------------------------ #
    def __call__(self, T: float | Iterable[float]) -> np.ndarray:
        return self.sa(T)

    def sd(self, T: float | Iterable[float]) -> np.ndarray:
        """Pseudo-displacement Sd = Sa * g * (T / 2 pi)^2, in metres."""
        T = self._periods(T)
        return self.sa(T) * G * (T / (2.0 * np.pi)) ** 2

    def default_periods(self, t_max: float = 4.0, n: int = 400) -> np.ndarray:
        """Sampling grid ``0 .. t_max`` that contains every knee period."""
        grid = np.linspace(0.0, t_max, n)
        knees = [k for k in self.knee_periods() if 0.0 < k <= t_max]
        return np.unique(np.concatenate([grid, knees]))

    def table(self, T: float | Iterable[float] | None = None) -> pd.DataFrame:
        """DataFrame with columns ``T``, ``Sa`` [g] and ``Sd`` [m]."""
        T = self.default_periods() if T is None else self._periods(T)
        return pd.DataFrame({"T": T, "Sa": self.sa(T), "Sd": self.sd(T)})

    # ------------------------------------------------------------------ #
    # Operations: each returns a new model (a DerivedSpectrum), so results
    # plot, export and combine like any code spectrum.  Sd of a derived
    # spectrum is the generic Sa g (T / 2 pi)^2.
    # ------------------------------------------------------------------ #
    def scaled(self, factor: float, *, label: str | None = None) -> "DerivedSpectrum":
        """Every ordinate times ``factor`` (> 0)."""
        f = _positive_factor("factor", factor)
        return DerivedSpectrum(label or f"{self.code} x {f:g}", lambda T: f * self.sa(T),
                               (self,), {"scale_factor": f})

    def reduced(
        self,
        R: float,
        *,
        Ie: float = 1.0,
        phi_p: float = 1.0,
        phi_e: float = 1.0,
        label: str | None = None,
    ) -> "DerivedSpectrum":
        """Inelastic design ordinates ``Sa Ie / (R phiP phiE)``.

        ASCE 7 uses ``R`` and ``Ie``; NEC-SE-DS uses ``R``, ``phiP``, ``phiE`` and
        ``I`` (pass it as ``Ie``).  For NEC static design prefer
        :meth:`NECSpectrum.reduced_sa`, which also drops the ascending branch.
        """
        vals = {n: _positive_factor(n, v) for n, v in
                (("R", R), ("Ie", Ie), ("phi_p", phi_p), ("phi_e", phi_e))}
        k = vals["Ie"] / (vals["R"] * vals["phi_p"] * vals["phi_e"])
        return DerivedSpectrum(label or f"{self.code} / R={vals['R']:g}",
                               lambda T: k * self.sa(T), (self,), vals)

    def envelope(self, *others: "CodeSpectrumModel", label: str | None = None) -> "DerivedSpectrum":
        """Point-wise maximum of this spectrum and ``others``."""
        models = (self, *others)
        if len(models) < 2:
            raise ValueError("envelope needs at least one other spectrum")
        return DerivedSpectrum(
            label or "envelope(" + ", ".join(m.code for m in models) + ")",
            lambda T: np.max([m.sa(T) for m in models], axis=0), models, {},
        )

    def floored_by(
        self, other: "CodeSpectrumModel", ratio: float = 1.0, *, label: str | None = None
    ) -> "DerivedSpectrum":
        """``max(Sa, ratio x other.Sa)`` at every period.

        E.g. the ASCE 7 Section 21.3 floor of a site-specific design spectrum at
        80 % of the Section 11.4 one: ``site.floored_by(code, 0.8)``.
        """
        r = _positive_factor("ratio", ratio)
        return DerivedSpectrum(label or f"{self.code} floored at {r:.0%} of {other.code}",
                               lambda T: np.maximum(self.sa(T), r * other.sa(T)),
                               (self, other), {"floor_ratio": r})

    def capped_by(
        self, other: "CodeSpectrumModel", ratio: float = 1.0, *, label: str | None = None
    ) -> "DerivedSpectrum":
        """``min(Sa, ratio x other.Sa)`` at every period."""
        r = _positive_factor("ratio", ratio)
        return DerivedSpectrum(label or f"{self.code} capped at {r:.0%} of {other.code}",
                               lambda T: np.minimum(self.sa(T), r * other.sa(T)),
                               (self, other), {"cap_ratio": r})

    def peak(self, t_max: float = 4.0, n: int = 2000) -> tuple[float, float]:
        """``(T, Sa)`` of the maximum ordinate on ``0..t_max`` (knees included)."""
        T = np.unique(np.concatenate([np.linspace(0.0, t_max, n), self.default_periods(t_max, 10)]))
        sa = self.sa(T)
        i = int(np.nanargmax(sa))
        return float(T[i]), float(sa[i])

    @staticmethod
    def _periods(T: float | Iterable[float]) -> np.ndarray:
        """Validate and return *T* as a 1-D float array (``>= 0``, finite)."""
        arr = np.atleast_1d(np.asarray(T, dtype=float))
        if arr.ndim != 1:
            raise ValueError("periods must be a scalar or a 1-D sequence")
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValueError("periods must be finite and >= 0")
        return arr


class AsceTwoPeriodSpectrum(CodeSpectrumModel):
    """Shape of the classical ASCE 7 two-period design response spectrum.

    Shared by ASCE 7-10, 7-16 and 7-22 (two-period procedure, not the
    multi-period spectra of 7-22).  A subclass computes ``sds``, ``sd1`` and
    ``tl`` (as plain attributes, normally in ``__post_init__``) from its own
    edition's site-coefficient tables; this class turns them into Sa(T):

    ====================  ==========================
    ``T < T0``            ``SDS (0.4 + 0.6 T / T0)``
    ``T0 <= T <= Ts``     ``SDS``
    ``Ts < T <= TL``      ``SD1 / T``
    ``T > TL``            ``SD1 TL / T^2``
    ====================  ==========================

    with ``Ts = SD1 / SDS`` and ``T0 = 0.2 Ts``.
    """

    sds: float
    sd1: float
    tl: float

    def _check_shape_inputs(self) -> None:
        if not (self.sds > 0.0 and self.sd1 > 0.0):
            raise ValueError("SDS and SD1 must be > 0")
        if not self.tl > self.ts:
            raise ValueError(f"TL ({self.tl}) must exceed Ts ({self.ts:.3f})")

    @property
    def ts(self) -> float:
        """Short/long-period corner ``Ts = SD1 / SDS`` [s]."""
        return self.sd1 / self.sds

    @property
    def t0(self) -> float:
        """Start of the constant plateau ``T0 = 0.2 Ts`` [s]."""
        return 0.2 * self.ts

    def knee_periods(self) -> tuple[float, ...]:
        return (self.t0, self.ts, self.tl)

    def sa(self, T: float | Iterable[float]) -> np.ndarray:
        T = self._periods(T)
        sds, sd1, tl, t0, ts = self.sds, self.sd1, self.tl, self.t0, self.ts
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                T < t0,
                sds * (0.4 + 0.6 * T / t0),
                np.where(
                    T <= ts,
                    sds,
                    np.where(T <= tl, sd1 / T, sd1 * tl / T**2),
                ),
            )


def _positive_factor(name: str, value: float) -> float:
    v = float(value)
    if not (np.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be a finite value > 0, got {value!r}")
    return v


class DerivedSpectrum(CodeSpectrumModel):
    """A spectrum computed from other models (scaled, reduced, envelope, floor, cap).

    Its ``code`` is a descriptive label; ``parameters()`` holds the operation's
    factors plus the parents' codes; ``knee_periods()`` is the union of the
    parents' knees, so sampling grids keep every corner.
    """

    code = "derived"

    def __init__(self, label: str, fn, parents: tuple[CodeSpectrumModel, ...],
                 params: dict[str, Any]) -> None:
        self.code = label  # type: ignore[misc]  (instance label overrides the ClassVar)
        self._fn = fn
        self.parents = tuple(parents)
        self._params = {**params, "from": ", ".join(p.code for p in self.parents)}

    def sa(self, T: float | Iterable[float]) -> np.ndarray:
        return np.asarray(self._fn(self._periods(T)), dtype=float)

    def parameters(self) -> dict[str, Any]:
        return dict(self._params)

    def knee_periods(self) -> tuple[float, ...]:
        return tuple(sorted({k for p in self.parents for k in p.knee_periods()}))
