"""Import / public-API smoke tests, including the filters __all__ regression."""
import importlib

import pytest


@pytest.mark.parametrize(
    "module, name",
    [
        ("apeQuake", "Record"),
        ("apeQuake.design", "asce7"),
        ("apeQuake.design.asce7", "base_shear"),
        ("apeQuake.design.nec", "base_shear"),
        ("apeQuake.code_spectrum", "DerivedSpectrum"),
        ("apeQuake.core", "Record"),
        ("apeQuake.filters", "Filter"),
        ("apeQuake.spectrum", "Spectrum"),
        ("apeQuake.spectrogram", "Spectrogram"),
        ("apeQuake.response_spectra", "ResponseSpectra"),
        ("apeQuake.intensity_measures", "IntensityMeasures"),
        ("apeQuake.code_spectrum", "CodeSpectrum"),
        ("apeQuake.plot_record", "PlotRecord"),
        ("apeQuake.hazard", "EcuadorHazard"),
        ("apeQuake.hazard", "HazardSite"),
    ],
)
def test_public_symbol_importable(module, name):
    mod = importlib.import_module(module)
    assert hasattr(mod, name), f"{module} is missing {name}"


@pytest.mark.parametrize(
    "module",
    [
        "apeQuake.design",
        "apeQuake.design.asce7",
        "apeQuake.design.nec",
        "apeQuake.filters",
        "apeQuake.spectrum",
        "apeQuake.spectrogram",
        "apeQuake.response_spectra",
        "apeQuake.intensity_measures",
        "apeQuake.code_spectrum",
        "apeQuake.plot_record",
        "apeQuake.hazard",
    ],
)
def test_all_entries_actually_exist(module):
    # Regression: filters.__all__ used to list "Filters" (wrong) instead of "Filter".
    mod = importlib.import_module(module)
    for name in getattr(mod, "__all__", []):
        assert hasattr(mod, name), f"{module}.__all__ lists missing symbol {name!r}"
