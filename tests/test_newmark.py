"""Golden-value validation of the Newmark SDOF solver and response-spectrum kernel.

The equation of motion solved by ResponseSpectra is (m = 1):

    u'' + 2 xi w u' + w^2 u = -a_g(t),   u(0) = u'(0) = 0

These tests pin the numerical result against:
  * an independent high-accuracy ODE integrator (scipy.solve_ivp),
  * the closed-form steady-state amplitude for harmonic excitation,
  * the pseudo-acceleration identity Sa = w^2 Sd,
  * agreement between the Numba kernel and the pure-Python fallback.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from apeQuake import Record
from apeQuake.response_spectra.response_spectra import ResponseSpectra


def test_newmark_matches_scipy_ode():
    """Newmark (avg-accel) must match an independent RK integrator for smooth input."""
    dt = 0.002
    t = np.arange(0.0, 12.0, dt)

    def ag_func(tt):
        # smooth, band-limited excitation
        return np.sin(2.0 * np.pi * 1.0 * tt) + 0.5 * np.sin(2.0 * np.pi * 2.3 * tt)

    ag = ag_func(t)
    T = 0.7
    xi = 0.05
    w = 2.0 * np.pi / T

    u, v, a = ResponseSpectra._newmark_sdof(ag=ag, dt=dt, T=T, damping=xi)

    def rhs(tt, y):
        u_, v_ = y
        return [v_, -ag_func(tt) - 2.0 * xi * w * v_ - w * w * u_]

    sol = solve_ivp(
        rhs, (t[0], t[-1]), [0.0, 0.0],
        t_eval=t, rtol=1e-10, atol=1e-12, max_step=dt,
    )
    u_ref, v_ref = sol.y[0], sol.y[1]

    peak_u = np.max(np.abs(u_ref))
    peak_v = np.max(np.abs(v_ref))
    assert np.max(np.abs(u - u_ref)) < 1e-3 * peak_u
    assert np.max(np.abs(v - v_ref)) < 1e-3 * peak_v


def test_harmonic_steady_state_amplitude():
    """For long harmonic forcing the relative-disp peak -> analytical Rd amplitude."""
    dt = 0.002
    t = np.arange(0.0, 60.0, dt)
    A = 1.0
    f_drive = 1.0
    Omega = 2.0 * np.pi * f_drive
    ag = A * np.sin(Omega * t)

    T = 0.9
    xi = 0.10
    w = 2.0 * np.pi / T
    beta = Omega / w

    u, v, a = ResponseSpectra._newmark_sdof(ag=ag, dt=dt, T=T, damping=xi)

    # steady-state amplitude of u'' + 2 xi w u' + w^2 u = -A sin(Omega t)
    Rd = 1.0 / np.sqrt((1.0 - beta**2) ** 2 + (2.0 * xi * beta) ** 2)
    u_amp_analytical = (A / w**2) * Rd

    # use the tail (transient decayed) to measure steady amplitude
    tail = u[t > 40.0]
    u_amp_numerical = np.max(np.abs(tail))

    assert np.isclose(u_amp_numerical, u_amp_analytical, rtol=0.02)


def test_pseudo_sa_equals_w2_sd():
    dt = 0.01
    t = np.arange(0.0, 20.0, dt)
    x = np.sin(2.0 * np.pi * 1.5 * t) + 0.3 * np.sin(2.0 * np.pi * 0.4 * t)
    rec = Record(x=x, dt=dt)

    T = np.linspace(0.1, 3.0, 40)
    rec.response_spectra.compute(periods=T, use_filters=False, sa_mode="pseudo")

    w = 2.0 * np.pi / T
    assert np.allclose(rec.response_spectra.Sa["X"], w**2 * rec.response_spectra.Sd["X"])


def test_numba_kernel_matches_python_fallback():
    dt = 0.01
    t = np.arange(0.0, 20.0, dt)
    x = np.sin(2.0 * np.pi * 1.5 * t) + 0.3 * np.sin(2.0 * np.pi * 0.4 * t)
    rec = Record(x=x, dt=dt)
    T = np.linspace(0.1, 3.0, 30)

    rec.response_spectra.compute(periods=T, use_filters=False, sa_mode="absolute", parallel=True)
    Sd_p, Sv_p, Sa_p = (rec.response_spectra.Sd["X"].copy(),
                        rec.response_spectra.Sv["X"].copy(),
                        rec.response_spectra.Sa["X"].copy())

    rec.response_spectra.compute(periods=T, use_filters=False, sa_mode="absolute", parallel=False)
    Sd_s, Sv_s, Sa_s = (rec.response_spectra.Sd["X"],
                        rec.response_spectra.Sv["X"],
                        rec.response_spectra.Sa["X"])

    assert np.allclose(Sd_p, Sd_s, rtol=1e-4, atol=1e-9)
    assert np.allclose(Sv_p, Sv_s, rtol=1e-4, atol=1e-9)
    assert np.allclose(Sa_p, Sa_s, rtol=1e-4, atol=1e-9)


def test_zero_motion_gives_zero_response():
    dt = 0.01
    rec = Record(x=np.zeros(2000), dt=dt)
    T = np.linspace(0.1, 3.0, 20)
    rec.response_spectra.compute(periods=T, use_filters=False)
    assert np.allclose(rec.response_spectra.Sd["X"], 0.0)
    assert np.allclose(rec.response_spectra.Sa["X"], 0.0)


def test_nonpositive_period_rejected():
    rec = Record(x=np.zeros(1000), dt=0.01)
    with pytest.raises(ValueError, match="periods must be > 0"):
        rec.response_spectra.compute(periods=[0.0, 1.0], use_filters=False)
