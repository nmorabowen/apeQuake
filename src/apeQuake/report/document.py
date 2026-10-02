"""Typst source of the site seismic report (informe register, ape-report template).

Prose follows the APE writing standard: impersonal, every parameter with its code,
edition and clause, every result as value, reference, ratio and one sentence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

NEC = "NEC-SE-DS 2015"
A16 = "ASCE/SEI 7-16"
A22 = "ASCE/SEI 7-22"
IGEPN = "IG-EPN"

CODES = (
    "Norma Ecuatoriana de la Construcción (NEC-2015), capítulo NEC-SE-DS (Peligro Sísmico, "
    "Diseño Sismo Resistente).",
    "American Society of Civil Engineers, Minimum Design Loads and Associated Criteria for "
    "Buildings and Other Structures (ASCE/SEI 7-16).",
    "American Society of Civil Engineers, Minimum Design Loads and Associated Criteria for "
    "Buildings and Other Structures (ASCE/SEI 7-22).",
    "Instituto Geofísico de la Escuela Politécnica Nacional (IG-EPN), Mapa Digital Interactivo "
    "del Peligro Sísmico Probabilístico para el Ecuador.",
)

ACRONYMS = (
    ("ASCE", "American Society of Civil Engineers"),
    ("IG-EPN", "Instituto Geofísico de la Escuela Politécnica Nacional"),
    ("MCE#sub[R]", "Sismo máximo considerado con riesgo objetivo"),
    ("NEC", "Norma Ecuatoriana de la Construcción"),
    ("UHS", "Espectro de peligro uniforme"),
)

NOTATION = (
    ("$Z$", "Factor de zona sísmica, aceleración máxima en roca [g]"),
    ("$eta$", "Razón de amplificación espectral (Sa/Z en roca)"),
    ("$F_a, F_d, F_s$", "Coeficientes de perfil de suelo, NEC-SE-DS 2015"),
    ("$S_S, S_1$", "Aceleración espectral en roca para T = 0.2 [s] y T = 1.0 [s] [g]"),
    ("$F_a, F_v$", "Coeficientes de sitio, ASCE/SEI 7-16"),
    ("$S_(D S), S_(D 1)$", "Aceleraciones espectrales de diseño [g]"),
    ("$T_C, T_L$", "Períodos límite del espectro NEC-SE-DS 2015 [s]"),
    ("$V_(s 30)$", "Velocidad media de onda de corte en los 30 m superiores [m/s]"),
)

_SPECIAL = set("\\#$*_@<>[]`~=/+-\"'")


def esc(text: Any) -> str:
    """Escape a value for Typst markup (names come from data, never trusted as markup)."""
    return "".join("\\" + ch if ch in _SPECIAL else ch for ch in str(text))


def f(x: float | None, n: int = 2) -> str:
    return "no aplica" if x is None else f"{x:.{n}f}"


def _cmp(value: float, ref: float) -> str:
    """'supera en 23 %' / 'queda 12 % por debajo' / 'coincide'."""
    p = 100.0 * (value / ref - 1.0)
    if abs(p) < 0.5:
        return "coincide con"
    return f"supera en {abs(p):.0f} % a" if p > 0 else f"queda {abs(p):.0f} % por debajo de"


@dataclass
class ReportMeta:
    """Front-matter data of the report."""

    title: str = "Parámetros Sísmicos del Sitio"
    subtitle: str = "NEC-SE-DS 2015, ASCE/SEI 7 e IG-EPN"
    client: str | None = None
    project: str | None = None
    document_code: str | None = None
    revision: str = "0"
    date: str | None = None
    site_name: str | None = None
    authors: list[dict[str, str]] = field(default_factory=list)
    revisions: list[dict[str, str]] | None = None   # default: one row for ``revision``


def _typ_str(s: str | None) -> str:
    if s is None:
        return "none"
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _front(meta: ReportMeta, abstract: str) -> str:
    # the cover template reads name, email and affiliation of every author
    authors = ", ".join(
        "(" + ", ".join(f"{k}: {_typ_str(a.get(k, ''))}" for k in ("name", "email", "affiliation")) + ",)"
        for a in meta.authors)
    codes = ", ".join(_typ_str(c) for c in CODES)
    rev_rows = meta.revisions if meta.revisions is not None else [
        {"rev": meta.revision, "date": meta.date or "", "description": "Emisión inicial",
         "prepared": "", "reviewed": "", "approved": ""}]
    revs = ", ".join(
        "(" + ", ".join(f"{k}: {_typ_str(v)}" for k, v in r.items()) + ",)" for r in rev_rows)
    acr = ", ".join(f"([{a}], {_typ_str(b)})" for a, b in ACRONYMS)
    nota = ", ".join(f"([{a}], {_typ_str(b)})" for a, b in NOTATION)
    return f"""#import "@local/ape-informes:0.1.0": *

#show: ape-report.with(
  title: {_typ_str(meta.title)},
  subtitle: {_typ_str(meta.subtitle)},
  client: {_typ_str(meta.client)},
  project: {_typ_str(meta.project)},
  document-code: {_typ_str(meta.document_code)},
  revision: {_typ_str(meta.revision)},
  date: {_typ_str(meta.date)},
  authors: ({authors}{"," if len(meta.authors) == 1 else ""}),
  revisions: ({revs}{"," if len(rev_rows) == 1 else ""}),
  show-abstract: true,
  abstract: [{abstract}],
  codes: ({codes},),
  acronyms: ({acr},),
  notation: ({nota},),
  bibliography-file: none,
)
"""


def _method_phrase(d: dict[str, Any]) -> str:
    m = d["ss_s1"]["method"]
    if m == "nec":
        return (f"a partir del espectro elástico de la {NEC} para perfil tipo B, "
                "multiplicado por 1.5 para llevar el nivel de diseño de 475 años al nivel "
                "del sismo máximo considerado")
    if m == "igepn":
        return (f"del espectro de peligro uniforme del {IGEPN} para 2475 años de período "
                "de retorno, en roca")
    return "por definición del ingeniero responsable"


def _where(d: dict[str, Any], meta: ReportMeta) -> str:
    z = d["zone"]
    name = f"{esc(meta.site_name)}, " if meta.site_name else ""
    return (f"{name}latitud {d['inputs']['lat']:.4f}°, longitud {d['inputs']['lon']:.4f}°, "
            f"provincia de {esc(z['province'].title())}")


def abstract(d: dict[str, Any], meta: ReportMeta) -> str:
    z, ss = d["zone"], d["ss_s1"]
    cls = d["site_classes"]
    rock = {r["T"]: r["rock"] for r in d["comparison"]}
    site = {r["T"]: r.get("site") for r in d["comparison"]}
    out = [
        f"Se determinan los parámetros de peligro sísmico del sitio ubicado en {_where(d, meta)}, "
        f"y se comparan las demandas espectrales de la {NEC}, del {A16}, del {A22} y del modelo "
        f"de peligro sísmico probabilístico del {IGEPN}.",
        "",
        f"El sitio se encuentra en la zona sísmica {z['zone']} de la {NEC} (§3.1.1, mapa de zonificación sísmica), "
        f"con factor de zona $Z = {f(z['z_used'])}$ [g] y razón de amplificación espectral "
        f"$eta = {f(z['eta_used'])}$ (región {esc(z['region_used'])}). Los parámetros en roca "
        f"adoptados son $S_S = {f(ss['ss'])}$ [g] y $S_1 = {f(ss['s1'])}$ [g], obtenidos "
        f"{_method_phrase(d)}.",
    ]
    if site[0.2] is not None:
        s2, s1 = site[0.2], site[1.0]

        def vals(s):
            parts = [f"{f(s['nec'])} [g] según la {NEC}"]
            if s["asce7_16"] is not None:
                parts.append(f"{f(s['asce7_16'])} [g] según el {A16}")
            if s["asce7_22"] is not None:
                parts.append(f"{f(s['asce7_22'])} [g] según el {A22}")
            return ", ".join(parts[:-1]) + " y " + parts[-1] if len(parts) > 1 else parts[0]

        out += ["",
                f"Para el perfil de suelo {cls['nec']} ({NEC}), la aceleración espectral de diseño "
                f"en $T = 0.2$ [s] es {vals(s2)}; en $T = 1.0$ [s] es {vals(s1)}."]
    r2 = rock[0.2]
    out += ["",
            f"En roca, el espectro del {IGEPN} para 475 años alcanza "
            f"$S_a = {f(r2['igepn_475'])}$ [g] en $T = 0.2$ [s], valor que {_cmp(r2['igepn_475'], r2['nec_475'])} "
            f"la ordenada de la {NEC} para perfil B, {f(r2['nec_475'])} [g]."]
    if d["warnings"]:
        n = len(d["warnings"])
        out += ["", (f"Los resultados se acompañan de una advertencia, listada" if n == 1 else
                     f"Los resultados se acompañan de {n} advertencias, listadas")
                + " en el capítulo de supuestos y limitaciones."]
    return "\n".join(out)


def introduction(d: dict[str, Any], meta: ReportMeta) -> str:
    return f"""= Introducción

== Objeto

El objeto del informe es establecer los parámetros sísmicos del sitio ubicado en {_where(d, meta)}, y cuantificar la diferencia entre las demandas espectrales de la {NEC}, del {A16}, del {A22} y del modelo de peligro sísmico probabilístico del {IGEPN}.

== Alcance

El alcance comprende la zonificación sísmica del sitio, la definición de los parámetros en roca $S_S$ y $S_1$, los espectros elásticos de aceleraciones en roca y para el perfil de suelo del proyecto, y la comparación de las ordenadas espectrales en $T = 0.2$ [s] y $T = 1.0$ [s]. El informe no incluye análisis de respuesta de sitio ni análisis de peligro sísmico específico del sitio; los casos en que la norma los exige se identifican en el capítulo de supuestos y limitaciones.

== Bases

- Zonificación sísmica y factor $Z$: {NEC} §3.1.1, Figura 1, digitalizada; contraste con la Tabla 19 (§10.2).
- Perfiles de suelo y coeficientes $F_a$, $F_d$ y $F_s$: {NEC} §3.2.1 y §3.2.2, Tablas 2 a 5.
- Espectro elástico de aceleraciones: {NEC} §3.3.1.
- Curvas de peligro sísmico de las capitales provinciales: {NEC} §3.1.2 y apéndice 10.3.
- Coeficientes de sitio $F_a$ y $F_v$ y espectro de diseño: {A16} §11.4.3 a §11.4.8, Tablas 11.4-1 y 11.4-2.
- Clases de sitio y espectro de dos períodos: {A22} Tabla 20.2-1 y §11.4.5.2.
- Peligro sísmico probabilístico en roca ($V_(s 30) = 760$ [m/s]) para 475 y 2475 años de período de retorno: {IGEPN}, Mapa Digital Interactivo del Peligro Sísmico Probabilístico para el Ecuador.

== Organización

El capítulo 2 presenta la ubicación y la zonificación sísmica del sitio; el capítulo 3, los parámetros en roca $S_S$ y $S_1$; el capítulo 4, los espectros de la {NEC}; el capítulo 5, los espectros del ASCE/SEI 7; el capítulo 6, el peligro sísmico probabilístico del {IGEPN}; el capítulo 7, la comparación de los espectros. Las conclusiones y los supuestos y limitaciones cierran el informe.
"""


def comparison_section(d: dict[str, Any]) -> str:
    cls = d["site_classes"]
    rows_rock, rows_site = [], []
    by_t = {r["T"]: r for r in d["comparison"]}
    r2, r1 = by_t[0.2]["rock"], by_t[1.0]["rock"]

    def row(label, a, b, ra, rb):
        cell = lambda v, ref: "no aplica" if v is None else f(v / ref)
        return (f'([{label}], [{f(a)}], [{f(b)}], [{cell(a, ra)}], [{cell(b, rb)}])')

    rows_rock = [
        row(f"{NEC}, perfil B", r2["nec_475"], r1["nec_475"], r2["nec_475"], r1["nec_475"]),
        row("ASCE/SEI 7, roca de referencia", r2["asce_design"], r1["asce_design"], r2["nec_475"], r1["nec_475"]),
        row(f"{IGEPN}, 475 años", r2["igepn_475"], r1["igepn_475"], r2["nec_475"], r1["nec_475"]),
        row(f"{IGEPN}, 2/3 × 2475 años", r2["igepn_2475_x2_3"], r1["igepn_2475_x2_3"], r2["nec_475"], r1["nec_475"]),
    ]
    rock_text = [
        f"La @fig-roca compara los espectros en roca al nivel de diseño: el espectro elástico de la "
        f"{NEC} para perfil B (475 años), el espectro de diseño del ASCE/SEI 7 para la roca de "
        f"referencia ($F_a = F_v = 1$, $S_(D S) = 2/3 S_S$, $S_(D 1) = 2/3 S_1$) y el espectro de "
        f"peligro uniforme del {IGEPN} para 475 años y para 2/3 de 2475 años."]
    if d["ss_s1"]["method"] == "nec":
        rock_text.append(
            f"Con $S_S$ y $S_1$ obtenidos de la {NEC}, los dos espectros normativos coinciden en "
            f"$T = 0.2$ [s] y $T = 1.0$ [s] por definición; difieren después de $T_L$, donde el "
            f"ASCE/SEI 7 decrece con $1 \\/ T^2$ y la {NEC} con $1 \\/ T$.")
    rock_text.append(
        f"En $T = 0.2$ [s] el espectro del {IGEPN} para 475 años, {f(r2['igepn_475'])} [g], "
        f"{_cmp(r2['igepn_475'], r2['nec_475'])} la ordenada de la {NEC}, {f(r2['nec_475'])} [g]; "
        f"en $T = 1.0$ [s], {f(r1['igepn_475'])} [g] {_cmp(r1['igepn_475'], r1['nec_475'])} "
        f"{f(r1['nec_475'])} [g].")

    parts = ["= Comparación de espectros", "", "== Espectros en roca", "", " ".join(rock_text), "",
             _figure("fig-roca.svg", "fig-roca",
                     f"Espectros de aceleraciones en roca, nivel de diseño. Fuente: {NEC} §3.3.1 y "
                     f"§3.1.2; ASCE/SEI 7 §11.4; {IGEPN}."),
             "",
             _table("tab-roca", "Ordenadas espectrales en roca y cociente frente a la "
                    f"{NEC}. Fuente: {NEC} §3.3.1; ASCE/SEI 7 §11.4; {IGEPN}.", rows_rock)]

    if d["site"] is not None:
        s2, s1 = by_t[0.2]["site"], by_t[1.0]["site"]
        rows_site = [
            row(f"{NEC}, perfil {cls['nec']}", s2["nec"], s1["nec"], s2["nec"], s1["nec"]),
            row(f"{A16}, clase {cls['asce7_16']}", s2["asce7_16"], s1["asce7_16"], s2["nec"], s1["nec"]),
            row(f"{A22}, clase {cls['asce7_22']} (aproximado)", s2["asce7_22"], s1["asce7_22"], s2["nec"], s1["nec"]),
            row(f"{IGEPN}, 475 años × amplificación {NEC}", s2["igepn_475_scaled"], s1["igepn_475_scaled"], s2["nec"], s1["nec"]),
        ]
        site_text = [
            f"La @fig-sitio compara los espectros de diseño para el perfil de suelo del proyecto. "
            f"Cada norma aplica sus propios coeficientes de sitio a los mismos $S_S$ y $S_1$: la "
            f"{NEC}, $F_a$, $F_d$ y $F_s$ (Tablas 3 a 5); el {A16}, $F_a$ y $F_v$ (Tablas 11.4-1 "
            f"y 11.4-2). El {A22} no tabula coeficientes de sitio: $S_(M S)$ y $S_(M 1)$ provienen "
            f"de la base de datos sísmica del USGS, que no cubre el Ecuador; se adoptan los valores "
            f"del {A16} con la forma espectral de dos períodos del {A22} (§11.4.5.2), por lo que "
            f"ese espectro es aproximado."]
        gov = _governing(s1)
        if gov:
            site_text.append(f"En $T = 1.0$ [s] gobierna {gov}.")
        parts += ["", "== Espectros para el perfil de suelo", "", " ".join(site_text), "",
                  _figure("fig-sitio.svg", "fig-sitio",
                          f"Espectros de diseño para el perfil de suelo del proyecto. Fuente: "
                          f"{NEC} §3.3.1; {A16} §11.4; {A22} §11.4.5.2; {IGEPN}."),
                  "",
                  _table("tab-sitio", "Ordenadas espectrales para el perfil de suelo y cociente "
                         f"frente a la {NEC}. Fuente: {NEC}; {A16}; {A22}; {IGEPN}.", rows_site)]
    return "\n".join(parts) + "\n"


def _governing(s: dict[str, Any]) -> str:
    cands = [(s["nec"], f"la {NEC}"), (s["asce7_16"], f"el {A16}"), (s["asce7_22"], f"el {A22}")]
    cands = [(v, n) for v, n in cands if v is not None]
    if len(cands) < 2:
        return ""
    v, n = max(cands)
    return f"{n} con {f(v)} [g], {f(v / s['nec'])} veces la ordenada de la {NEC}"


def _figure(path: str, label: str, caption: str, width: str = "100%") -> str:
    return f'#figure(image("{path}", width: {width}), caption: [{caption}]) <{label}>'


def _table(label: str, caption: str, rows: list[str]) -> str:
    return (f"#tabla-ape(\n  caption: [{caption}],\n  columns: (1fr, auto, auto, auto, auto),\n"
            f"  align: (left, center, center, center, center),\n"
            f'  encabezado: ("Fuente", "Sa(0.2 s) [g]", "Sa(1.0 s) [g]", "Cociente 0.2 s", '
            f'"Cociente 1.0 s"),\n  filas: (\n    ' + ",\n    ".join(rows) + ",\n  ),\n) <" + label + ">")


def render(d: dict[str, Any], meta: ReportMeta) -> str:
    """Full Typst source for the assessment dict ``d`` (``SiteAssessment.to_dict()``)."""
    from . import chapters as ch

    return "\n".join([
        _front(meta, abstract(d, meta)), introduction(d, meta),
        ch.ubicacion(d, _where(d, meta)), ch.parametros_roca(d), ch.espectros_nec(d),
        ch.espectros_asce(d), ch.peligro_igepn(d), comparison_section(d),
        ch.conclusiones(d), ch.supuestos(d),
    ])
