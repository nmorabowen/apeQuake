# NEC hazard curves

NEC-SE-DS (NEC-15), section 10.3, plots the seismic hazard of 23 provincial
capitals (Figures 10–32): the annual exceedance rate of **PGA** and of
**Sa at T = 0.1, 0.2, 0.5 and 1.0 s** against acceleration. They only exist as
raster images in the PDF, so `apeQuake.nec` ships them as numbers, digitized
from the official document, and adds interpolation, **extrapolation**, return
period inversion and a uniform-hazard spectrum.

```python
from apeQuake.nec import load_hazard_database

db = load_hazard_database()
quito = db["Quito"]                  # case / accent insensitive: db["tulcán"]

quito["PGA"].a_at_return_period(475)     # array([0.436])  g
quito["0.2"].rate_at(0.8)                # annual exceedance rate at 0.8 g
quito.hazard_table()                     # PGA / Sa at the four NEC levels
```

## The four NEC hazard levels

`hazard_table()` evaluates every measure at the NEC levels (Table 9): 72, 225,
475 and 2 500 years.

```python
quito.hazard_table(tail="quadratic")
#                          PGA  Sa(0.1s)  Sa(0.2s)  Sa(0.5s)  Sa(1s)
# frecuente (Tr=72)       ...
# ocasional (Tr=225)      ...
# raro (Tr=475)           ...
# muy raro (Tr=2500)      ...
```

Per the NEC (§4.1), forces taken from these curves are **not** multiplied by
the importance factor `I`.

## Uniform-hazard spectrum

```python
periods, sa = quito.uhs(return_period=475)   # T = 0, 0.1, 0.2, 0.5, 1.0 s
```

Only five periods exist in the source plots, so the shape between them is a
straight line; use it as a check against the NEC design spectrum, not as a
replacement for it.

## Extrapolation

The plots stop at 10⁻⁵ 1/yr or at 0.8–2.0 g, whichever comes first. Rarer
events (for example Tr = 10 000 yr) lie beyond them. Past the plotted range,
`rate_at`, `a_at_rate` and `a_at_return_period` use a **log-log quadratic tail
model**

$$\ln\lambda(a) = c_0 + c_1 \ln a + c_2 (\ln a)^2$$

fitted to the last 30 % of each curve (`tail_fraction`). If the fit would turn
upward it falls back to the end slope. A pure power law (`tail="linear"`) is
available but *not recommended*: hazard curves bend steeper in log-log.

```python
cur = quito["1.0"]                       # leaves the plot at ~1.05 g, 1e-5 1/yr
cur.a_at_rate(1e-6)                      # beyond the plot: extrapolated
cur.a_at_return_period(1e4, extrapolate=False)   # NaN outside the data
longer = cur.extrapolate(3.0)            # a new HazardCurve to 3 g
quito.plot(extrapolate_to=2.5)           # extrapolation drawn dashed
```

### How good is it?

Back-test: hide each curve below a cutoff rate, fit on the rest, predict the
acceleration at a lower rate and compare with the digitized value (all
cities and periods):

| Hidden range | Quadratic tail, mean error | 90th percentile | Power-law tail, mean error |
|---|---|---|---|
| 10⁻³ → 10⁻⁴ (one decade) | 3 % | 6 % | 20 % |
| 10⁻³ → 3·10⁻⁵ (1.5 decades) | 7 % | 11 % | 41 % |

Extrapolating further than about 1.5 decades beyond the data is a judgement
call. Treat those numbers as order-of-magnitude.

## Accuracy and caveats

* **Digitization.** The source is a raster plot: expect about ±2–4 % in annual
  rate (1–2 px on a five-decade axis). The pipeline is reproducible with
  `scripts/digitize_nec_hazard.py`, which also writes overlay images for visual
  QA.
* **Hidden curves.** Where one curve is drawn over another, the hidden stretch
  follows the curve on top (it lies underneath it in the plot). Sa(0.2 s) is
  almost entirely hidden under Sa(0.1 s) in some cities; `curve.aliased_from`
  flags a curve that is a copy.
* **The curves do not reproduce the zone factor Z everywhere.** The PGA at
  Tr = 475 yr read from these plots is close to Z in the Sierra centre and
  Guayaquil (Quito 0.44 vs Z = 0.40, Guayaquil 0.40 vs 0.40), above the
  saturated Z = 0.50 on the coast (Esmeraldas 0.97), but well *below* Z in the
  north and Oriente (Tulcán 0.25 vs 0.40, Orellana 0.13 vs 0.25). This is how the
  NEC figures read, not a digitizing artefact. Use Z for the NEC design
  spectrum and these curves only where the NEC asks for them (multiple
  hazard levels, essential structures, bridges, ports).
* **Rock motion.** The curves are for rock, 5 % damping; apply site
  coefficients Fa, Fd, Fs separately.
