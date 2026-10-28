"""Record-file readers (PEER AT2, RENAC, generic text) and Record.from_files.

The fixture files below reproduce the headers of real files: PEER NGA
``RSN125_FRIULI.A_A-TMZ000.AT2`` and RENAC ``ACH1_201604162359_E_100.txt``
(Latin-1, as distributed).
"""
import numpy as np
import pytest

from apeQuake import Record
from apeQuake.core.units import G, conversion_factor, normalize_unit
from apeQuake.readers import detect_format, read_at2, read_record_file, read_renac, read_txt

VALUES = np.round(np.sin(np.linspace(0.0, 6.0, 23)) * 1e-2, 7)


def _rows(values, per_row):
    return "\n".join(
        " ".join(f"{v: .7E}" for v in values[i:i + per_row]) for i in range(0, len(values), per_row)
    )


def write_at2(path, values=VALUES, dt=0.005, old_layout=False):
    line4 = (
        f"{len(values):7d}   {dt:.4f}    NPTS, DT"
        if old_layout
        else f"NPTS=   {len(values)}, DT=   {dt:.4f} SEC,"
    )
    path.write_text(
        "PEER NGA STRONG MOTION DATABASE RECORD\n"
        "Friuli Italy-01, 5/6/1976, Tolmezzo, 0\n"
        "ACCELERATION TIME SERIES IN UNITS OF G\n"
        f"{line4}\n{_rows(values, 5)}\n"
    )
    return path


def write_renac(path, values=VALUES * 100.0, freq=100.0, component="E"):
    header = (
        "Archivo de aceleración parte de la Red Nacional de Acelerógrafos (RENAC) Ecuador\n"
        "          \n"
        "Evento: 201604162358\n"
        "Fecha del evento UTM (aammdd):                  2016    4   16\n"
        "Hora del registro UTM (hhmmss): 23   59 45.00\n"
        "Estación:                                      ACH1 \n"
        f"Componente:                                        {component}\n"
        f"Frecuencia de muestreo (Hz):            {freq:.6e}\n"
        "Unidades: cm/s^2\n"
        "_____________________________________________\n"
    )
    path.write_bytes((header + _rows(values, 6) + "\n").encode("latin-1"))
    return path


def test_at2_new_layout(tmp_path):
    f = read_at2(write_at2(tmp_path / "a.AT2"))
    assert f.format == "at2" and f.units == "g"
    assert f.dt == 0.005 and f.npts == VALUES.size
    np.testing.assert_allclose(f.data, VALUES, rtol=1e-6)
    assert f.metadata["title"].startswith("Friuli")


def test_at2_old_layout(tmp_path):
    f = read_at2(write_at2(tmp_path / "a.at2", dt=0.01, old_layout=True))
    assert f.dt == 0.01 and f.npts == VALUES.size


def test_at2_trailing_padding_is_cut_to_npts(tmp_path):
    p = write_at2(tmp_path / "a.AT2")
    p.write_text(p.read_text() + "  .0000000E+00  .0000000E+00\n")
    assert read_at2(p).npts == VALUES.size


def test_at2_short_data_raises(tmp_path):
    p = write_at2(tmp_path / "a.AT2")
    p.write_text(p.read_text().replace(f"NPTS=   {VALUES.size}", "NPTS=   999"))
    with pytest.raises(ValueError, match="NPTS=999"):
        read_at2(p)


def test_renac_latin1_header(tmp_path):
    f = read_renac(write_renac(tmp_path / "ACH1_E.txt"))
    assert f.format == "renac" and f.units == "cm/s2"
    assert np.isclose(f.dt, 0.01)
    assert f.metadata["station"] == "ACH1"
    assert f.metadata["component"] == "E"
    assert f.metadata["event"] == "201604162358"  # not the "Fecha del evento" line
    np.testing.assert_allclose(f.data, VALUES * 100.0, rtol=1e-6)


def test_renac_malformed_line_raises_not_zero_filled(tmp_path):
    p = write_renac(tmp_path / "r.txt")
    p.write_bytes(p.read_bytes() + b"0.1 0.2 ### 0.4\n")
    with pytest.raises(ValueError, match="non-numeric data on line"):
        read_renac(p)


def test_txt_dt_from_header_and_argument(tmp_path):
    p = tmp_path / "v.txt"
    p.write_text("Station XYZ vertical\nUnits: m/s^2\nDT = 0.02\n" + "\n".join(map(str, VALUES)))
    f = read_txt(p)
    assert f.dt == 0.02 and f.units == "m/s2"
    np.testing.assert_allclose(f.data, VALUES)
    with pytest.raises(ValueError, match="disagrees"):
        read_txt(p, dt=0.01)

    q = tmp_path / "bare.txt"
    q.write_text("\n".join(map(str, VALUES)))
    with pytest.raises(ValueError, match="pass dt="):
        read_txt(q)
    assert read_txt(q, dt=0.01).units is None


def test_detect_format(tmp_path):
    assert detect_format(write_at2(tmp_path / "a.AT2")) == "at2"
    assert detect_format(write_renac(tmp_path / "r.txt")) == "renac"
    q = tmp_path / "plain.txt"
    q.write_text("DT=0.01\n1 2 3\n")
    assert detect_format(q) == "txt"
    assert read_record_file(q).npts == 3


def test_units():
    assert normalize_unit("Gal") == "cm/s2"
    assert normalize_unit("m/s^2") == "m/s2"
    assert np.isclose(conversion_factor("g", "cm/s2"), 100.0 * G)
    with pytest.raises(ValueError):
        normalize_unit("ft/s2")


def test_from_files_renac_three_components(tmp_path):
    files = {c: write_renac(tmp_path / f"ACH1_{c}.txt", component=c) for c in "ENZ"}
    rec = Record.from_files(x=files["E"], y=files["N"], z=files["Z"])
    assert rec.components == ("X", "Y", "Z")
    assert rec.units == "cm/s2"
    assert np.isclose(rec.dt, 0.01)
    assert rec.name == "ACH1 201604162358"
    assert set(rec.source_files) == {"X", "Y", "Z"}
    np.testing.assert_allclose(rec.x, VALUES * 100.0, rtol=1e-6)


def test_from_files_converts_and_pads(tmp_path):
    at2 = write_at2(tmp_path / "a.AT2", dt=0.01)
    renac = write_renac(tmp_path / "r.txt", values=VALUES[:-3] * 100.0)
    with pytest.raises(ValueError, match="different units"):
        Record.from_files(x=at2, y=renac)

    rec = Record.from_files(x=at2, y=renac, units="cm/s2")
    assert rec.units == "cm/s2"
    np.testing.assert_allclose(rec.x, VALUES * 100.0 * G, rtol=1e-6)
    assert rec.y[-3:].tolist() == [0.0, 0.0, 0.0]
    assert any("padded with 3 trailing zeros" in s for s in rec.init_logs)
    assert any("converted g -> cm/s2" in s for s in rec.init_logs)


def test_from_files_rejects_mixed_dt(tmp_path):
    with pytest.raises(ValueError, match="different time steps"):
        Record.from_files(x=write_at2(tmp_path / "a.AT2", dt=0.005),
                          y=write_at2(tmp_path / "b.AT2", dt=0.01))


def test_from_files_unknown_units_need_units_argument(tmp_path):
    q = tmp_path / "bare.txt"
    q.write_text("\n".join(map(str, VALUES)))
    rec = Record.from_files(x=q, dt=0.01)
    assert rec.units is None
    with pytest.raises(ValueError, match="declare no unit"):
        Record.from_files(x=q, y=write_at2(tmp_path / "a.AT2", dt=0.01), dt=0.01)
    assert Record.from_files(x=q, dt=0.01, units="g").units == "g"
