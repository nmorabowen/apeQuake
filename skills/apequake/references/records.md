# Records and code spectra

## `Record`

```python
from apeQuake import Record
rec = Record(x=ax, y=ay, z=az, dt=0.01, name="Station")   # any of x / y / z

rec.plot_record.plot()
rec.spectrum.apply_base_filters(detrend=True, demean=True, taper=0.05)
rec.spectrum.compute(); rec.spectrum.plot(representation="loglog")

rec.response_spectra.add_filter(rec.filter.detrend, type="demean")   # pipeline declared once
rec.response_spectra.add_filter(rec.filter.taper, max_percentage=0.05)
rec.response_spectra.compute(periods=np.linspace(0.05, 5, 200))
rec.response_spectra.plot(quantity="Sa", representation="loglog", combined=True)

rec.intensity_measures.significant_duration("X", p1=0.05, p2=0.95)
rec.intensity_measures.husid("X")
rec.spectrogram.compute(); rec.spectrogram.plot_spectrogram()
```

Composites live on the record (`filter`, `spectrum`, `response_spectra`, `spectrogram`,
`intensity_measures`, `plot_record`, `code_spectrum`); each processes every component. Guides:
`docs/guides/*.md`.

## Code spectra (`apeQuake.code_spectrum`)

```python
from apeQuake.code_spectrum.codes.nec import NECSpectrum, ETA_BY_REGION, site_coefficients
from apeQuake.code_spectrum.codes.asce7_16 import ASCE7_16Spectrum
from apeQuake.code_spectrum.codes.asce7_22 import ASCE7_22Spectrum

NECSpectrum(z=0.4, site_class="D", region="sierra").sa(T)      # .parameters(): Z, eta, Fa, Fd, Fs, r, T0, Tc, TL
ASCE7_16Spectrum(ss, s1, "D", tl, allow_exception=True)          # .fa .fv .sms .sm1 .sds .sd1 .exceptions
ASCE7_16Spectrum.from_sds_sd1(sds, sd1, tl)                      # site-specific / reference rock
ASCE7_22Spectrum(sms, sm1, tl, "CD")                             # SMS/SM1 inputs (no Fa/Fv in 7-22)
rec.code_spectrum.add("ASCE7-16", ...)                           # overlay on a record's spectrum
```

- NEC: Fa/Fd/Fs interpolated in Z (0.15-0.50), F raises; TL = 2.4·Fd capped at 4 s for D/E.
- ASCE 7-16: §11.4.8 triggers raise unless `allow_exception=True`; class E with S1 > 0.1 raises
  regardless (no Fv in Table 11.4-2).
