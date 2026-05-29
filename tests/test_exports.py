"""Import / public-API smoke tests, including the filters __all__ regression."""
import importlib

import pytest


@pytest.mark.parametrize(
    "module, name",
    [
        ("apeQuake", "Record"),
        ("apeQuake.core", "Record"),
        ("apeQuake.filters", "Filter"),
        ("apeQuake.spectrum", "Spectrum"),
        ("apeQuake.spectrogram", "Spectrogram"),
        ("apeQuake.response_spectra", "ResponseSpectra"),
        ("apeQuake.intensity_measures", "IntensityMeasures"),
        ("apeQuake.plot_record", "PlotRecord"),
    ],
)
def test_public_symbol_importable(module, name):
    mod = importlib.import_module(module)
    assert hasattr(mod, name), f"{module} is missing {name}"


@pytest.mark.parametrize(
    "module",
    [
        "apeQuake.filters",
        "apeQuake.spectrum",
        "apeQuake.spectrogram",
        "apeQuake.response_spectra",
        "apeQuake.intensity_measures",
        "apeQuake.plot_record",
    ],
)
def test_all_entries_actually_exist(module):
    # Regression: filters.__all__ used to list "Filters" (wrong) instead of "Filter".
    mod = importlib.import_module(module)
    for name in getattr(mod, "__all__", []):
        assert hasattr(mod, name), f"{module}.__all__ lists missing symbol {name!r}"
