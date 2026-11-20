"""Evaluacion de los pronosticos: precision relativa al paseo aleatorio y encompassing.

Objetivos 2017-01 a 2026-08 sin dic-2023 (115 por horizonte). Los pronosticos de los modelos de comparacion (RW, VAR,
BVAR, FAVAR, TVP-VAR, TVP-VAR-SV, RF, REM) se leen de resultados/01_pronosticos; el GTVP-VAR y el MRF salen de
pronosticos_gtvp.py y mrf.py.
- MAE y RMSE relativos al paseo aleatorio, con el p-valor de Giacomini-White (perdida absoluta y cuadratica);
- GTVP-VAR contra cada modelo;
- encompassing entre GTVP-VAR y TVP-VAR (Harvey, Leybourne y Newbold, 1998), con perdida cuadratica y absoluta.
Salidas: resultados/02_evaluacion/{evaluacion_final.md, encompassing.md, errores_final.csv}
Uso: python -m ejemplos.inflacion_argentina.evaluacion
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import config as cf
from .pruebas import estrellas, gw

PRON = cf.RESULTADOS / "01_pronosticos"
SALIDA = cf.RESULTADOS / "02_evaluacion"
ORDEN = ["GTVP-VAR", "TVP-VAR", "TVP-VAR-SV", "BVAR", "VAR", "FAVAR", "VAR pequeño", "RF", "MRF", "REM"]


def errores():
    """modelo -> serie de errores (variacion simple) indexada por (h, objetivo)."""
    def f(a, modelo=None):
        x = pd.read_csv(PRON / a, parse_dates=["objetivo"])
        if modelo is not None:
            x = x[x.modelo == modelo]
        x = x[(x.h <= cf.H) & x.objetivo.between(cf.INI, cf.FIN)]
        return x.set_index(["h", "objetivo"]).error.sort_index()
    E = {"GTVP-VAR": f("pronosticos_gtvp_final_h2.csv"), "TVP-VAR": f("pronosticos_tvp_kk.csv"),
         "TVP-VAR-SV": f("pronosticos_tvp_sv.csv")}
    for m in ["RW", "BVAR", "VAR", "FAVAR", "VAR pequeño"]:
        E[m] = f("pronosticos_lineales_final.csv", m)
    E["RF"] = f("pronosticos_rf_hiper_gtvp.csv")
    E["MRF"] = f("pronosticos_mrf_hiper.csv")
    r = pd.read_csv(PRON / "pronosticos_rem.csv", parse_dates=["objetivo"])
    r = r[r.objetivo.between(cf.INI, cf.FIN)]
    r["error"] = r.pronostico - r.objetivo.map(cf.observado())
    E["REM"] = r.set_index(["h", "objetivo"]).error.sort_index()
    return E


def comparar(a, b, h, absoluta):
    a, b = a.loc[h], b.loc[h]
    idx = a.index.intersection(b.index)
    a, b = a.reindex(idx).values, b.reindex(idx).values
    k = np.asarray(idx != cf.FUERA)
    L = np.abs if absoluta else np.square
    rel = L(a[k]).mean() / L(b[k]).mean()
    rel = rel if absoluta else np.sqrt(rel)
    return rel, gw(a, b, h, k, absoluta), int(k.sum())


def tabla_evaluacion(E):
    md = ["# Comparación final de modelos", "",
          "Objetivos 2017-01 a 2026-08 sin dic-23. Todos los modelos con dic-23 fuera de la estimación. "
          "Entre paréntesis, p-valor de Giacomini-White contra el benchmark de cada tabla.", ""]
    for absoluta, nom in ((True, "MAE"), (False, "RMSE")):
        md += [f"## {nom} relativo al RW", "", "| Modelo | " + " | ".join(f"h={h}" for h in range(1, cf.H + 1)) + " | n |",
               "|---|" + "---:|" * (cf.H + 1)]
        for m in ORDEN:
            cel, n = [], 0
            for h in range(1, cf.H + 1):
                r, p, n = comparar(E[m], E["RW"], h, absoluta)
                cel.append(f"{r:.3f}{estrellas(p)} ({p:.3f})")
            md.append(f"| {m} | " + " | ".join(cel) + f" | {n} |")
        md.append("")
    md += ["## GTVP-VAR contra cada modelo (MAE del GTVP-VAR / MAE del modelo)", "",
           "| Modelo | " + " | ".join(f"h={h}" for h in range(1, cf.H + 1)) + " |", "|---|" + "---:|" * cf.H]
    for m in ["RW"] + ORDEN[1:]:
        cel = []
        for h in range(1, cf.H + 1):
            r, p, _ = comparar(E["GTVP-VAR"], E[m], h, True)
            cel.append(f"{r:.3f}{estrellas(p)} ({p:.3f})")
        md.append(f"| {m} | " + " | ".join(cel) + " |")
    return "\n".join(md)


def _t_hac(d, mascara, h):
    n, mu = mascara.sum(), d[mascara].mean()
    u = np.where(mascara, d - mu, 0.0)
    om = u @ u / n
    for L in range(1, h):
        om += 2 * (1 - L / h) * (u[L:] @ u[:-L]) / n
    return mu / np.sqrt(om / n), n


def tabla_encompassing():
    """H0: A encompassa a B (B no agrega informacion); una cola. Perdida cuadratica: d_t = e_A (e_A - e_B), HLN con
    correccion de muestra chica contra t(n - 1) (y Clark-West contra la normal). Perdida absoluta:
    d_t = signo(e_A) (e_A - e_B) contra la normal; lambda = peso optimo de B (LAD)."""
    arch = {"GTVP-VAR": "pronosticos_gtvp_final_h2.csv", "TVP-VAR": "pronosticos_tvp_kk.csv"}
    E = {m: pd.read_csv(PRON / a, parse_dates=["objetivo"]).query("objetivo >= @cf.INI and objetivo <= @cf.FIN")
         .set_index(["h", "objetivo"]).error * 100 for m, a in arch.items()}
    filas = []
    for A, B in [("GTVP-VAR", "TVP-VAR"), ("TVP-VAR", "GTVP-VAR")]:
        for h in range(1, cf.H + 1):
            ea, eb = E[A].loc[h].sort_index(), E[B].loc[h].sort_index()
            assert (ea.index == eb.index).all()
            mascara = np.asarray(ea.index != cf.FUERA)
            ea, eb = ea.values, eb.values
            d = ea * (ea - eb)
            t, n = _t_hac(d, mascara, h)
            hln = t * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
            lam = d[mascara].sum() / ((ea - eb)[mascara] ** 2).sum()
            dd = (ea - eb)[mascara]
            ok = dd != 0
            r, w = ea[mascara][ok] / dd[ok], np.abs(dd[ok])
            o = np.argsort(r)
            lam_a = r[o][np.searchsorted(np.cumsum(w[o]), w.sum() / 2)]
            ta, _ = _t_hac(np.sign(ea) * (ea - eb), mascara, h)
            filas.append(dict(A=A, B=B, h=h, lam=lam, p_hln=stats.t.sf(hln, n - 1), p_cw=stats.norm.sf(t),
                              lam_a=lam_a, p_abs=stats.norm.sf(ta)))
    R = pd.DataFrame(filas)
    md = ["# Encompassing de pronósticos: GTVP-VAR y TVP-VAR", "",
          "Escenario dic-23 fuera de la estimación; GTVP-VAR con la regla (1000 árboles), TVP-VAR con prior de Koop y "
          "Korobilis. Objetivos 2017-01 a 2026-08 sin dic-23 (115 por h). H0: A encompassa a B (B no agrega "
          "información); H1 de una cola: B agrega. λ: peso óptimo de B en la combinación (1 − λ)·A + λ·B. "
          "No anidados: Harvey, Leybourne y Newbold (1998). Anidados: Clark y West (2007). * 10%, ** 5%, *** 1%.", ""]
    for (A, B), g in R.groupby(["A", "B"], sort=False):
        md += [f"## ¿{B} agrega información a {A}?", "",
               "| | " + " | ".join(f"h={h}" for h in g.h) + " |", "|---|" + "---:|" * len(g),
               "| λ (peso de " + B + ") | " + " | ".join(f"{v:.2f}" for v in g.lam) + " |",
               "| p no anidados (HLN) | " + " | ".join(f"{p:.3f}{estrellas(p)}" for p in g.p_hln) + " |",
               "| p anidados (Clark-West) | " + " | ".join(f"{p:.3f}{estrellas(p)}" for p in g.p_cw) + " |",
               "| λ con pérdida absoluta (LAD) | " + " | ".join(f"{v:.2f}" for v in g.lam_a) + " |",
               "| p con pérdida absoluta | " + " | ".join(f"{p:.3f}{estrellas(p)}" for p in g.p_abs) + " |", ""]
    return "\n".join(md)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    E = errores()
    pd.concat({m: e for m, e in E.items()}, names=["modelo"]).rename("error").reset_index().to_csv(
        SALIDA / "errores_final.csv", index=False)
    for nombre, txt in (("evaluacion_final.md", tabla_evaluacion(E)), ("encompassing.md", tabla_encompassing())):
        (SALIDA / nombre).write_text(txt, encoding="utf-8")
        print(txt, "\n")


if __name__ == "__main__":
    main()
