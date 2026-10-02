# Site assessment: `apeQuake.site_assessment.assess_site`

Merged in apeQuake#11 (2026-10-01).

```python
from apeQuake.site_assessment import assess_site, site_class_nec, site_class_asce7_22, PERIODS

a = assess_site(lat, lon, *, vs30=None, site_class=None,      # exactly one of the two
                method="nec",                                   # "nec" | "igepn" | "manual"
                ss=None, s1=None,                               # manual only
                z=None, region=None,                            # overrides of the map
                tl_asce=None)                                   # default: NEC TL of the class
d = a.to_dict()       # strict JSON (NaN/inf -> None)
```

## Result fields (`SiteAssessment`)

| Field | Content |
|---|---|
| `zone` | `zone_at(...).to_dict()` + `z_used`, `region_used`, `eta_used` |
| `site_classes` | `{"nec", "asce7_16", "asce7_22"}` (Vs30 → each code's table; class → same letter) |
| `ss_s1` | `method`, `ss`, `s1`, `candidates` {nec, igepn[, manual]}, `nec_rule` {factor, sa_b_02, sa_b_10} |
| `nec` | `parameters` (Z, eta, Fa, Fd, Fs, r, T0, Tc, TL, ...) / `rock_parameters`, `spectrum` / `rock_spectrum` |
| `nec_uhs` | nearest digitized capital: `city`, `distance_km`, `"475"`, `"2500"` curves |
| `asce7_16` | `parameters` (Fa, Fv, SMS, SM1, SDS, SD1, T0, Ts, TL), `exceptions`, `tl_source`, `spectrum`; `None` for F or E with S1 > 0.1 |
| `asce7_22` | `parameters` (+ `Fa_7_16`, `Fv_7_16`), `approximate: True`, `note`, `spectrum`; `None` like 7-16 |
| `igepn` | `cell_id`, `distance_km`, `periods` (8), `uhs` {"475","2475"} × {mean, q16, q84} |
| `rock` | curves `nec_475`, `asce_design`, `igepn_475`, `igepn_2475_x2_3`, `nec_uhs_475` |
| `site` | curves `nec`, `asce7_16`, `asce7_22`, `igepn_475_scaled`, `igepn_2475_x2_3_scaled`, `nec_amplification`; `None` for F |
| `comparison` | rows at T = 0.2 and 1.0 s with `rock` and `site` values |
| `warnings` | zoning + IG-EPN + ASCE 11.4.8 + overrides + far NEC city (> 25 km), English text |
| `notices` | the same as `Notice(code, text, params)` (`apeQuake.notices`); `asce_unavailable.reason` = `class_e_no_fv` or `other` |

Curves are `{"T": [...], "Sa": [...]}`; code spectra on `PERIODS` (0-4 s every 0.02 s + 0.05, 0.07).

## Rules

- Levels: NEC elastic (475 yr); ASCE design = 2/3 MCE_R; IG-EPN 475 and 2/3 × 2475.
- Rock view: NEC class B; ASCE reference rock `from_sds_sd1(2/3 Ss, 2/3 S1)` (Fa = Fv = 1).
  With method `nec` the two coincide at 0.2 and 1.0 s by construction; they split after TL
  (ASCE 1/T², NEC 1/T).
- Site view: IG-EPN scaled by `Sa_NEC,site(T) / Sa_NEC,B(T)` (approximate).
- Vs30 tables: NEC / 7-16 A ≥ 1500, B ≥ 760, C ≥ 360, D ≥ 180, E; 7-22 (m/s, from ft/s)
  A > 1524, B > 914.4, BC > 640.1, C > 442, CD > 304.8, D > 213.4, DE > 152.4, E.
- Class B from the dropdown: `vs_measured=False` → Fa = Fv = 1 (ASCE 7-16 §11.4.3); with Vs30
  the measured-B values 0.9 / 0.8 apply.
- ASCE 7-16: `allow_exception=True` always; triggers in `exceptions` (`D_S1`, `E_Ss`, `E_S1`).

## Reference numbers (tests)

- Quito, class D, method nec: Ss = 1.488, S1 = 0.614; Sa(1.0 s): NEC 0.83, 7-16 0.99 (11.4.8
  shape), 7-22 0.70, IG-EPN scaled 0.59. IG-EPN rock 2/3·2475 at 0.2 s = 1.42 (NEC 0.99).
- Office sheet `NEC_SE_DS - ASCE 7-16 (2016 v.3).xlsm` (Dropbox, Rumipamba): manual Ss = 2.1,
  S1 = 0.7, D → Fa 1.0, Fv 1.7, SDS 1.40, SD1 0.793 (also `test_code_spectrum_office_sheet.py`).
