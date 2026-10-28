"""Export of code spectra: CSV, ETABS / SAP2000 function files, OpenSees Path series."""
import re
import sys
import types

import numpy as np
import pandas as pd
import pytest

from apeQuake import Record
from apeQuake.code_spectrum import (
    MPRS_PERIODS,
    ASCE7_22MultiPeriodSpectrum,
    G,
    get_model_class,
    to_csv,
    to_etabs,
    to_opensees,
    to_sap2000,
)
from apeQuake.code_spectrum.export import sampled

ASCE16 = get_model_class("ASCE7-16")(ss=1.5, s1=0.6, site_class="C", tl=8.0)
SA = np.linspace(0.5, 0.1, len(MPRS_PERIODS))
MPRS = ASCE7_22MultiPeriodSpectrum(MPRS_PERIODS, SA, 12.0, "D")


def test_sampled_keeps_knees_and_unique_periods():
    T, sa = sampled(ASCE16, t_max=10.0, n=50)
    assert np.all(np.diff(T) > 0.0)
    for k in ASCE16.knee_periods():
        assert np.any(np.isclose(T, round(k, 6)))
    np.testing.assert_allclose(sa, ASCE16.sa(T))
    T2, sa2 = sampled(ASCE16, [0.5, 0.5, 1.0, 0.1], units="cm/s2")
    np.testing.assert_allclose(T2, [0.1, 0.5, 1.0])
    np.testing.assert_allclose(sa2, ASCE16.sa(T2) * 100.0 * G)
    with pytest.raises(ValueError, match="acceleration unit"):
        sampled(ASCE16, units="ft/s2")


def test_units_come_from_core_units(tmp_path):
    from apeQuake.code_spectrum.base import G as G_base
    from apeQuake.core.units import G as G_core

    assert G_base is G_core
    _, gal = sampled(ASCE16, [1.0], units="Gal")
    _, cms2 = sampled(ASCE16, [1.0], units="cm/s2")
    np.testing.assert_allclose(gal, cms2)
    head = to_csv(ASCE16, tmp_path / "s.csv", [1.0], units="m/s^2").read_text().splitlines()[0]
    assert head == "T [s],Sa [m/s2]"


def test_csv(tmp_path):
    df = pd.read_csv(to_csv(ASCE16, tmp_path / "s.csv", units="m/s2"))
    assert list(df.columns) == ["T [s]", "Sa [m/s2]"]
    np.testing.assert_allclose(df["Sa [m/s2]"], ASCE16.sa(df["T [s]"]) * G, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("writer", [to_etabs, to_sap2000])
def test_etabs_sap2000_header_skip_line_is_correct(writer, tmp_path):
    p = writer(MPRS, tmp_path / "f.txt", t_max=12.0)
    lines = p.read_text().splitlines()
    skip = int(re.fullmatch(r"Header lines to skip: (\d+)", lines[0]).group(1))
    data = np.loadtxt(p, skiprows=skip)
    T, sa = sampled(MPRS, t_max=12.0)
    np.testing.assert_allclose(data[:, 0], T)
    np.testing.assert_allclose(data[:, 1], sa, atol=1e-6)
    assert any("SDS=" in s for s in lines[:skip])
    assert np.any(np.isclose(data[:, 0], 10.0))     # last MPRS node kept


def test_opensees_tcl(tmp_path):
    txt = to_opensees(ASCE16, tmp_path / "rs.tcl", tag=7).read_text()
    m = re.search(r"timeSeries Path 7 -time \{([^}]*)\} -values \{([^}]*)\}", txt)
    T = np.array(m.group(1).split(), float)
    sa = np.array(m.group(2).split(), float)
    assert T.size == sa.size and np.all(np.diff(T) > 0.0)
    np.testing.assert_allclose(sa, ASCE16.sa(T), atol=1e-6)


def test_opensees_python_is_valid_openseespy(tmp_path, monkeypatch):
    calls = []
    ops = types.ModuleType("openseespy.opensees")
    ops.timeSeries = lambda *a: calls.append(a)
    pkg = types.ModuleType("openseespy")
    pkg.opensees = ops
    monkeypatch.setitem(sys.modules, "openseespy", pkg)
    monkeypatch.setitem(sys.modules, "openseespy.opensees", ops)

    p = to_opensees(ASCE16, tmp_path / "rs.py", style="python", units="m/s2", tag=3)
    exec(compile(p.read_text(), str(p), "exec"), {})  # noqa: S102 - our own generated file
    (args,) = calls
    assert args[:3] == ("Path", 3, "-time") and args[4] == "-values"
    T, sa = np.array(args[3]), np.array(args[5])
    assert all(type(v) is float for v in args[3])
    np.testing.assert_allclose(sa, ASCE16.sa(T) * G, atol=1e-6)


def test_opensees_validation(tmp_path):
    with pytest.raises(ValueError, match="style"):
        to_opensees(ASCE16, tmp_path / "x", style="matlab")
    with pytest.raises(ValueError, match="tag"):
        to_opensees(ASCE16, tmp_path / "x", tag=0)


def test_composite_export(tmp_path):
    rec = Record(x=np.zeros(10), dt=0.01)
    rec.code_spectrum.add("NEC-15", z=0.4, site_class="D")
    out = rec.code_spectrum.export("NEC-15", tmp_path / "nec.txt", units="cm/s2")
    assert out.read_text().startswith("Header lines to skip:")
    assert rec.code_spectrum.export("NEC-15", tmp_path / "nec.tcl", format="opensees").exists()
    with pytest.raises(ValueError, match="format"):
        rec.code_spectrum.export("NEC-15", tmp_path / "x", format="xlsx")
    with pytest.raises(KeyError):
        rec.code_spectrum.export("ASCE7-16", tmp_path / "x")
