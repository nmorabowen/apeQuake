"""Body chapters of the site report (informe register of the APE writing standard).

Every number comes from the assessment dict ``d`` (``SiteAssessment.to_dict()``);
data strings go through ``esc``.  Equations are typeset with their substitution.
"""
from __future__ import annotations

from typing import Any

from .document import A16, A22, IGEPN, NEC, _cmp, _figure, esc, f

REGION = {"costa": "Costa", "sierra": "Sierra", "oriente": "Oriente",
          "esmeraldas": "Esmeraldas", "galapagos": "Galápagos"}
TRIGGER = {"D_S1": "perfil D con $S_1 >= 0.2$", "E_Ss": "perfil E con $S_S >= 1.0$",
           "E_S1": "perfil E con $S_1 >= 0.2$"}
PERIOD_LABEL = {0.0: "PGA", 0.05: "0.05", 0.07: "0.07", 0.1: "0.10", 0.2: "0.20",
                0.5: "0.50", 1.0: "1.00", 2.0: "2.00"}


def tabla(label: str, caption: str, header: list[str], rows: list[list[str]],
          columns: str | None = None, align: str | None = None) -> str:
    """``tabla-ape`` call; cells are Typst markup already escaped."""
    n = len(header)
    columns = columns or "(" + ", ".join(["1fr"] + ["auto"] * (n - 1)) + ")"
    align = align or "(" + ", ".join(["left"] + ["center"] * (n - 1)) + ")"
    head = ", ".join(f"[{h}]" for h in header)
    body = ",\n    ".join("(" + ", ".join(f"[{c}]" for c in r) + ")" for r in rows)
    return (f"#tabla-ape(\n  caption: [{caption}],\n  columns: {columns},\n  align: {align},\n"
            f"  encabezado: ({head},),\n  filas: (\n    {body},\n  ),\n) <{label}>")


def _sa_at(curve: dict[str, Any], t: float) -> float:
    return float(curve["Sa"][[round(x, 4) for x in curve["T"]].index(round(t, 4))])


# ------------------------------------------------------------------- notices
def notice_es(n: dict[str, Any]) -> str:
    """A notice of apeQuake in Spanish (Typst markup); unknown codes keep their text."""
    p = n.get("params", {})
    code = n.get("code")
    if code == "near_boundary":
        return (f"El sitio está a {f(p['boundaryKm'], 1)} [km] del límite con la zona de "
                f"$Z = {f(p['zAcross'])}$ [g]; el mapa digitalizado tiene una resolución de "
                "1.5 [km], por lo que el valor se contrasta con la Tabla 19.")
    if code == "table19_differs":
        return (f"La Tabla 19 de la {NEC} asigna $Z = {f(p['zTable'])}$ [g] a "
                f"{esc(p['town'])} ({esc(p['canton'])}), a {f(p['distanceKm'], 1)} [km]; el mapa "
                f"de zonificación da $Z = {f(p['zMap'])}$ [g]. Para las poblaciones listadas, la "
                "norma adopta el valor de la tabla.")
    if code == "zona_no_delimitada":
        return "El sitio está en una zona no delimitada; se adopta la región Costa, $eta = 1.80$."
    if code == "site_class_f":
        return (f"Perfil de suelo F: la {NEC} (§3.2) y el ASCE/SEI 7 (§11.4.7 y §21.1) exigen un "
                "análisis de respuesta de sitio; el informe presenta solo la comparación en roca.")
    if code == "z_override":
        return (f"Se adopta $Z = {f(p['z'])}$ [g], definido por el ingeniero responsable; el mapa "
                f"de zonificación da $Z = {f(p['zMap'])}$ [g].")
    if code == "region_override":
        return (f"Se adopta la región {REGION.get(p['region'], esc(p['region']))}, definida por el "
                f"ingeniero responsable; la provincia corresponde a "
                f"{REGION.get(p['regionMap'], esc(p['regionMap']))}.")
    if code == "igepn_note":
        return f"{IGEPN}: {esc(p['message'])}"
    if code == "nec_city_far":
        return (f"Las curvas de peligro de la {NEC} existen solo para las capitales provinciales; "
                f"se usa {esc(p['city'])}, a {f(p['distanceKm'], 1)} [km].")
    if code == "asce_unavailable":
        if p.get("reason") == "class_e_no_fv":
            return (f"La Tabla 11.4-2 del {A16} no define $F_v$ para el perfil E con $S_1 > 0.1$ "
                    "(§11.4.8): se requiere un análisis de peligro específico del sitio y el "
                    "informe no presenta espectros ASCE/SEI 7 para el perfil.")
        return f"El informe no presenta espectros ASCE/SEI 7 para el perfil: {esc(p.get('message', ''))}"
    if code == "asce716_trigger":
        t = "; ".join(TRIGGER.get(x, esc(x)) for x in p.get("triggers", []))
        return (f"La §11.4.8 del {A16} exige un análisis de peligro específico del sitio ({t}); "
                "se aplica la excepción de la misma sección.")
    if code == "asce716_exception3":
        return (f"La excepción 3 de la §11.4.8 del {A16} (perfil E, $S_1 >= 0.2$) solo es válida "
                "para $T <= T_s$ con el método de la fuerza lateral equivalente.")
    return esc(n.get("text", ""))


# ------------------------------------------------------------------ chapters
def ubicacion(d: dict[str, Any], where: str) -> str:
    z = d["zone"]
    listed = z["nearest_listed"]
    out = ["= Ubicación y zonificación sísmica", "",
           f"El sitio se ubica en {where}. De acuerdo con la sección 3.1.1 de la {NEC}, el factor "
           "de zona $Z$ se obtiene del mapa de zonificación sísmica de la norma (Figura 1), "
           "digitalizado con una resolución de 1.5 [km] (@fig-ubicacion). El sitio pertenece a la "
           f"zona sísmica {z['zone']}, con $Z = {f(z['z'])}$ [g]."]
    if z["boundary_km"] is None:
        out[-1] += " No existe otra zona a menos de 20 [km]."
    else:
        out[-1] += (f" El límite más cercano con otra zona, de $Z = {f(z['z_across_boundary'])}$ [g], "
                    f"está a {f(z['boundary_km'], 1)} [km].")
    if listed is not None:
        same = abs(listed["z"] - z["z"]) < 1e-9
        out += ["", f"La población más cercana de la Tabla 19 (§10.2) es {esc(listed['town'])} "
                    f"({esc(listed['canton'])}), a {f(listed['distance_km'], 1)} [km], con "
                    f"$Z = {f(listed['z'])}$ [g]; "
                    + ("el valor coincide con el del mapa." if same else
                       "el valor difiere del mapa. Para las poblaciones listadas, la norma adopta "
                       "el valor de la tabla.")]
    out += ["", f"La razón de amplificación espectral $eta$ depende de la provincia (§3.3.1): el "
                f"sitio está en la provincia de {esc(z['province'].title())}, región "
                f"{REGION.get(z['region'], esc(z['region']))}, con $eta = {f(z['eta'])}$."]
    if abs(z["z_used"] - z["z"]) > 1e-9 or z["region_used"] != z["region"]:
        out += ["", f"Para este informe se adopta $Z = {f(z['z_used'])}$ [g] y la región "
                    f"{REGION.get(z['region_used'], esc(z['region_used']))} ($eta = {f(z['eta_used'])}$), "
                    "definidos por el ingeniero responsable."]
    out += ["", _figure("fig-ubicacion.svg", "fig-ubicacion",
                        f"Ubicación del sitio sobre el mapa de zonificación sísmica digitalizado. "
                        f"Fuente: {NEC} §3.1.1, Figura 1.", width="70%"), ""]
    rows = [["Factor de zona $Z$", f"{f(z['z_used'])} [g]"], ["Zona sísmica", z["zone"]],
            ["Provincia", esc(z["province"].title())],
            ["Región, $eta$", f"{REGION.get(z['region_used'], esc(z['region_used']))}, {f(z['eta_used'])}"],
            ["Distancia al límite de zona", "más de 20 [km]" if z["boundary_km"] is None
             else f"{f(z['boundary_km'], 1)} [km] (Z = {f(z['z_across_boundary'])} [g])"]]
    if listed is not None:
        rows.append(["Población de la Tabla 19 más cercana",
                     f"{esc(listed['town'])}, Z = {f(listed['z'])} [g], {f(listed['distance_km'], 1)} [km]"])
    out.append(tabla("tab-zona", f"Zonificación sísmica del sitio. Fuente: {NEC} §3.1.1, §3.3.1 y Tabla 19.",
                     ["Parámetro", "Valor"], rows, columns="(1fr, 1fr)", align="(left, left)"))
    return "\n".join(out) + "\n"


def parametros_roca(d: dict[str, Any]) -> str:
    ss = d["ss_s1"]
    z = d["zone"]
    out = ["= Parámetros en roca", "",
           "Los parámetros $S_S$ y $S_1$ se definen una sola vez, en roca de referencia "
           "($V_(s 30) = 760$ [m/s]), y cada norma aplica sobre ellos sus propios coeficientes de "
           "sitio.", ""]
    if ss["method"] == "nec":
        r = ss["nec_rule"]
        out += [f"Se obtienen del espectro elástico de la {NEC} para perfil tipo B "
                f"($F_a = F_d = 1.0$) con $Z = {f(z['z_used'])}$ [g] y $eta = {f(z['eta_used'])}$:", "",
                f"$ S_S = 1.5 dot S_(a,B)(0.2 \"s\") = 1.5 dot {f(r['sa_b_02'], 3)} = {f(ss['ss'], 3)} \"[g]\" $", "",
                f"$ S_1 = 1.5 dot S_(a,B)(1.0 \"s\") = 1.5 dot {f(r['sa_b_10'], 3)} = {f(ss['s1'], 3)} \"[g]\" $", "",
                "El factor 1.5 lleva el nivel de diseño de 475 años al nivel del sismo máximo "
                "considerado, en correspondencia con $S_D = 2/3 S_M$ del ASCE/SEI 7. Los valores no "
                "tienen riesgo objetivo."]
    elif ss["method"] == "igepn":
        out += [f"$S_S = {f(ss['ss'], 3)}$ [g] y $S_1 = {f(ss['s1'], 3)}$ [g] son las ordenadas en "
                f"$T = 0.2$ [s] y $T = 1.0$ [s] del espectro de peligro uniforme medio del {IGEPN} para "
                "2475 años de período de retorno, en roca. Los valores no tienen riesgo objetivo."]
    else:
        out += [f"$S_S = {f(ss['ss'], 3)}$ [g] y $S_1 = {f(ss['s1'], 3)}$ [g], definidos por el "
                "ingeniero responsable."]
    labels = {"nec": f"{NEC}, 1.5 × perfil B", "igepn": f"{IGEPN}, 2475 años", "manual": "Definidos por el ingeniero"}
    rows = [[labels[k] + (" (adoptado)" if k == ss["method"] else ""), f(v["ss"], 3), f(v["s1"], 3)]
            for k, v in ss["candidates"].items()]
    out += ["", tabla("tab-ss-s1", "Parámetros en roca por método. Fuente: "
                      f"{NEC} §3.3.1; {IGEPN}.", ["Método", "$S_S$ [g]", "$S_1$ [g]"], rows)]
    return "\n".join(out) + "\n"


def espectros_nec(d: dict[str, Any]) -> str:
    rock, site = d["nec"]["rock_parameters"], d["nec"]["parameters"]
    out = ["= Espectros de la NEC-SE-DS 2015", "", '#txt-espectro-intro(fuente: "nec")', ""]
    rows = []
    for label, p in (("Roca, perfil B", rock), (f"Perfil {site['site_class']}" if site else None, site)):
        if p is None:
            continue
        rows.append([label, f(p["Fa"], 3), f(p["Fd"], 3), f(p["Fs"], 3), f(p["r"], 1),
                     f(p["T0"], 3), f(p["Tc"], 3), f(p["TL"], 3), f(p["eta"] * p["Z"] * p["Fa"], 3)])
    out.append(tabla("tab-nec", f"Coeficientes de perfil de suelo y períodos límite. Fuente: {NEC} "
                     "§3.2.2, Tablas 3 a 5, y §3.3.1.",
                     ["Perfil", "$F_a$", "$F_d$", "$F_s$", "$r$", "$T_0$ [s]", "$T_C$ [s]",
                      "$T_L$ [s]", "$eta Z F_a$ [g]"], rows))
    if site is None:
        out += ["", "El perfil F requiere un análisis de respuesta de sitio (§3.2); el informe "
                    "presenta solo el espectro en roca."]
    u = d["nec_uhs"]
    sa02 = _sa_at(u["475"], 0.2)
    plateau = rock["eta"] * rock["Z"] * rock["Fa"]
    out += ["", f"Las curvas de peligro sísmico de la §3.1.2 se encuentran digitalizadas para las "
                f"capitales provinciales; la más cercana al sitio es {esc(u['city'])}, a "
                f"{f(u['distance_km'], 1)} [km]. Para 475 años su espectro de peligro uniforme da "
                f"$S_a = {f(sa02, 3)}$ [g] en $T = 0.2$ [s], valor que {_cmp(sa02, plateau)} la meseta "
                f"del espectro elástico en roca, $eta Z F_a = {f(plateau, 3)}$ [g]."]
    return "\n".join(out) + "\n"


def espectros_asce(d: dict[str, Any]) -> str:
    a16, a22, cls, ss = d["asce7_16"], d["asce7_22"], d["site_classes"], d["ss_s1"]
    out = ["= Espectros del ASCE/SEI 7", "", f"== {A16}", "",
           f"El {A16} obtiene los parámetros de diseño a partir de $S_S$ y $S_1$ con los "
           "coeficientes de sitio $F_a$ y $F_v$ de las Tablas 11.4-1 y 11.4-2: "
           "$S_(M S) = F_a S_S$ y $S_(M 1) = F_v S_1$ (Ec. 11.4-1 y 11.4-2), "
           "$S_(D S) = 2/3 S_(M S)$ y $S_(D 1) = 2/3 S_(M 1)$ (Ec. 11.4-3 y 11.4-4)."]
    if a16 is None:
        out += ["", f"Para la clase de sitio {cls['asce7_16']} el {A16} no define coeficientes de sitio "
                    "aplicables a los parámetros del sitio; el informe no presenta espectros "
                    "ASCE/SEI 7 para el perfil (capítulo de supuestos y limitaciones)."]
        return "\n".join(out) + "\n"
    p = a16["parameters"]
    out += ["", f"Para la clase de sitio {cls['asce7_16']}:", "",
            f"$ S_(M S) = {f(p['Fa'], 3)} dot {f(ss['ss'], 3)} = {f(p['SMS'], 3)} \"[g]\", quad "
            f"S_(M 1) = {f(p['Fv'], 3)} dot {f(ss['s1'], 3)} = {f(p['SM1'], 3)} \"[g]\" $", "",
            f"$ S_(D S) = {f(p['SDS'], 3)} \"[g]\", quad S_(D 1) = {f(p['SD1'], 3)} \"[g]\", quad "
            f"T_s = {f(p['Ts'], 3)} \"s\" $", "",
            f"El valor de $T_L = {f(p['TL'], 3)}$ [s] se adopta igual al $T_L$ de la {NEC} para el "
            "mismo perfil" + (" por definición del ingeniero responsable." if a16["tl_source"] == "input"
                              else "; el ASCE/SEI 7 no publica un mapa de $T_L$ para el Ecuador.")]
    ex = a16["exceptions"]
    if "D_S1" in ex:
        out += ["", "Para el perfil D con $S_1 >= 0.2$ la §11.4.8 exige un análisis de peligro "
                    "específico del sitio. Se aplica la excepción 2 de la misma sección: "
                    "$S_a = S_(D S)$ hasta $1.5 T_s$, y $1.5 S_(D 1) \\/ T$ a partir de ese período."]
    if "E_Ss" in ex:
        out += ["", "Para el perfil E con $S_S >= 1.0$ se aplica la excepción 1 de la §11.4.8: "
                    "$F_a$ igual al del perfil C."]
    if "E_S1" in ex:
        out += ["", "Para el perfil E con $S_1 >= 0.2$ la excepción 3 de la §11.4.8 solo es válida "
                    "para $T <= T_s$ con el método de la fuerza lateral equivalente."]
    out += ["", f"== {A22}", "",
            f"El {A22} (§11.4.3) toma $S_(M S)$ y $S_(M 1)$ directamente de la base de datos sísmica "
            "del USGS para cada clase de sitio y no tabula coeficientes de sitio; esa base no cubre "
            f"el Ecuador. Se adoptan $S_(M S)$ y $S_(M 1)$ del {A16} con la clase de sitio "
            f"{cls['asce7_22']} (Tabla 20.2-1) y el espectro de dos períodos de la §11.4.5.2, sin la "
            f"excepción de la §11.4.8 del {A16}. El espectro resultante es aproximado.", ""]
    rows = [[A16 + f", clase {cls['asce7_16']}", f(p["Fa"], 3), f(p["Fv"], 3), f(p["SMS"], 3),
             f(p["SM1"], 3), f(p["SDS"], 3), f(p["SD1"], 3), f(p["Ts"], 3), f(p["TL"], 3)]]
    if a22 is not None:
        q = a22["parameters"]
        rows.append([A22 + f", clase {cls['asce7_22']} (aproximado)", f(q["Fa_7_16"], 3),
                     f(q["Fv_7_16"], 3), f(q["SMS"], 3), f(q["SM1"], 3), f(q["SDS"], 3),
                     f(q["SD1"], 3), f(q["Ts"], 3), f(q["TL"], 3)])
    out.append(tabla("tab-asce", f"Coeficientes de sitio y parámetros de diseño. Fuente: {A16} "
                     f"§11.4, Tablas 11.4-1 y 11.4-2; {A22} §11.4.5.2 y Tabla 20.2-1.",
                     ["Norma", "$F_a$", "$F_v$", "$S_(M S)$", "$S_(M 1)$", "$S_(D S)$", "$S_(D 1)$",
                      "$T_s$ [s]", "$T_L$ [s]"], rows))
    return "\n".join(out) + "\n"


def peligro_igepn(d: dict[str, Any]) -> str:
    ig = d["igepn"]
    T = ig["periods"]
    u475, u2475 = ig["uhs"]["475"], ig["uhs"]["2475"]
    rows = [[PERIOD_LABEL.get(round(t, 2), f(t)), f(u475["mean"][i], 3), f(u475["q16"][i], 3),
             f(u475["q84"][i], 3), f(u2475["mean"][i], 3), f(u2475["q16"][i], 3),
             f(u2475["q84"][i], 3)] for i, t in enumerate(T)]
    i02 = [round(t, 2) for t in T].index(0.2)
    out = ["= Peligro sísmico probabilístico del IG-EPN", "",
           f"El modelo de peligro sísmico probabilístico del {IGEPN} entrega espectros de peligro "
           "uniforme en roca ($V_(s 30) = 760$ [m/s]) para 475 y 2475 años de período de retorno, con "
           "la media y los fractiles del árbol lógico, en celdas de 0.08°. El sitio corresponde a la "
           f"celda {esc(ig['cell_id'])}, con valores interpolados entre las celdas vecinas "
           f"(@tab-igepn).", "",
           f"Para 2475 años, la ordenada media en $T = 0.2$ [s] es {f(u2475['mean'][i02], 3)} [g], con "
           f"fractiles del 16 % y 84 % de {f(u2475['q16'][i02], 3)} [g] y {f(u2475['q84'][i02], 3)} [g].",
           "",
           tabla("tab-igepn", f"Espectros de peligro uniforme del {IGEPN} en roca [g]. Fuente: {IGEPN}, "
                 "Mapa Digital Interactivo del Peligro Sísmico Probabilístico para el Ecuador.",
                 ["$T$ [s]", "475 media", "475 q16", "475 q84", "2475 media", "2475 q16", "2475 q84"],
                 rows)]
    return "\n".join(out) + "\n"


def conclusiones(d: dict[str, Any]) -> str:
    z, ss = d["zone"], d["ss_s1"]
    rock = {r["T"]: r["rock"] for r in d["comparison"]}
    site = {r["T"]: r.get("site") for r in d["comparison"]}
    out = ["= Conclusiones", "",
           f"+ El sitio pertenece a la zona sísmica {z['zone']} de la {NEC}, con "
           f"$Z = {f(z['z_used'])}$ [g] y $eta = {f(z['eta_used'])}$."]
    if any(n.get("code") == "table19_differs" for n in d.get("notices", [])):
        out[-1] += " La Tabla 19 asigna otro valor a una población cercana; rige el valor de la tabla para las poblaciones listadas."
    out.append(f"+ Los parámetros en roca adoptados son $S_S = {f(ss['ss'], 3)}$ [g] y "
               f"$S_1 = {f(ss['s1'], 3)}$ [g].")
    for t in (0.2, 1.0):
        r = rock[t]
        out.append(f"+ En roca y $T = {f(t, 1)}$ [s], el espectro del {IGEPN} para 475 años, "
                   f"{f(r['igepn_475'], 3)} [g], {_cmp(r['igepn_475'], r['nec_475'])} la ordenada de la "
                   f"{NEC}, {f(r['nec_475'], 3)} [g].")
    if site[0.2] is not None:
        for t in (0.2, 1.0):
            s = site[t]
            cands = [(s["nec"], f"la {NEC}"), (s["asce7_16"], f"el {A16}"),
                     (s["asce7_22"], f"el {A22} (aproximado)")]
            cands = [(v, n) for v, n in cands if v is not None]
            v, n = max(cands)
            ratio = f(v / s["nec"])
            out.append(f"+ Para el perfil de suelo y $T = {f(t, 1)}$ [s] gobierna {n} con {f(v, 3)} [g], "
                       f"{ratio} veces la ordenada de la {NEC}.")
    else:
        out.append("+ El perfil F requiere un análisis de respuesta de sitio; no se comparan espectros para el perfil.")
    return "\n".join(out) + "\n"


def supuestos(d: dict[str, Any]) -> str:
    out = ["= Supuestos y limitaciones", "",
           f"- El factor de zona proviene del mapa de la {NEC} digitalizado con una resolución de "
           "1.5 [km]. El mapa coincide con la Tabla 19 en el 91 % de las poblaciones listadas; las "
           "diferencias se concentran junto a los límites de zona y en poblaciones donde la tabla y "
           "el mapa de la norma asignan valores distintos.",
           "- $S_S$ y $S_1$ no son valores con riesgo objetivo ($\"MCE\"_R$); provienen del método "
           "indicado en el capítulo de parámetros en roca.",
           f"- El espectro del {A22} es aproximado: usa $S_(M S)$ y $S_(M 1)$ obtenidos con las tablas "
           f"del {A16}, porque la base de datos sísmica del USGS no cubre el Ecuador.",
           f"- El espectro del {IGEPN} para el perfil de suelo se obtiene escalando el espectro en roca "
           f"con la amplificación de la {NEC}, $S_(a,\"perfil\")(T) \\/ S_(a,B)(T)$; es una aproximación.",
           f"- $T_L$ del ASCE/SEI 7 se adopta igual al de la {NEC} para el mismo perfil.",
           "- El informe no incluye análisis de respuesta de sitio ni análisis de peligro sísmico "
           "específico del sitio."]
    notices = d.get("notices", [])
    if notices:
        out += ["", "Advertencias del cálculo:", ""] + [f"+ {notice_es(n)}" for n in notices]
    return "\n".join(out) + "\n"
