"""Pronosticos pseudo fuera de muestra del GTVP-VAR (seccion 4 de la tesis).

En cada origen (2016-07 a 2026-07) se estima un bosque de 1000 arboles con la ventana de 272 pares que termina en el
origen (sin el par con objetivo dic-2023) y los hiperparametros elegidos en el julio anterior; se pronostica iterando
h = 1..6 con la media de las matrices de los arboles. Se repite con 3 semillas ([N, 500, origen, 3]); el pronostico
final usa el bosque combinado (los 3000 arboles).
Salidas: resultados/01_pronosticos/pronosticos_gtvp_final_h2{,_sem1,_sem2,_sem3}.csv,
         resultados/02_evaluacion/gtvp_final_h2.md; bosques en trabajo/bosques_gtvp/sem{N}/ (se reutilizan)
Uso: python -m replicacion.pronosticos_gtvp
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from gtvpvar import GTVPVAR

from . import config as cf
from .gtvp import Contexto, hp_dict
from .pruebas import estrellas, gw

SEMILLAS = (1, 2, 3)
SALIDA = cf.RESULTADOS / "01_pronosticos"


def carpeta(sem):
    return cf.TRABAJO / "bosques_gtvp" / f"sem{sem}"


def origen(ctx, i, hp, sem):
    arch = carpeta(sem) / f"origen_{i}.npz"
    if arch.exists():
        m = GTVPVAR.cargar(arch)
    else:
        m = ctx.ajustar(i, [sem, 500, i, 3], **hp_dict(hp))
        m.guardar(arch)
    return ctx.estado.iterar(ctx.beta_media(m), i, ctx.S, ctx.Z)


def combinado(ctx, i):
    ms = [GTVPVAR.cargar(carpeta(s) / f"origen_{i}.npz") for s in SEMILLAS]
    return ctx.estado.iterar(ctx.beta_media(*ms), i, ctx.S, ctx.Z)


def guardar(ctx, res, nombre, modelo):
    obs = cf.observado()
    fe = ctx.fe
    df = pd.DataFrame([dict(modelo=modelo, origen=fe[i], h=h, objetivo=fe[i + h], pronostico=np.expm1(z[0]))
                       for r in res for i, h, z in r if i + h < len(fe)])
    df["observado"] = df.objetivo.map(obs)
    df["error"] = df.pronostico - df.observado
    df.to_csv(SALIDA / nombre, index=False)
    return df


def tabla(P):
    """MAE / RW y p-valor GW (perdida absoluta), objetivos 2017-01 a 2026-08 sin dic-2023."""
    obs = cf.observado()
    md = ["# GTVP-VAR: semillas y bosque combinado", "",
          "MAE / RW (p-valor GW, perdida absoluta), objetivos 2017-01 a 2026-08 sin dic-2023.", "",
          "| | " + " | ".join(f"h={h}" for h in range(1, cf.H + 1)) + " |", "|---|" + "---:|" * cf.H]
    for nom, x in P.items():
        cel = []
        for h in range(1, cf.H + 1):
            g = x[(x.h == h) & x.objetivo.between(cf.INI, cf.FIN)].sort_values("objetivo")
            e = g.error.values
            rw = obs.loc[g.origen].values - g.objetivo.map(obs).values
            k = (g.objetivo != cf.FUERA).values
            p = gw(e, rw, h, k)
            cel.append(f"{np.abs(e[k]).mean() / np.abs(rw[k]).mean():.3f}{estrellas(p)} ({p:.3f})")
        md.append(f"| {nom} | " + " | ".join(cel) + " |")
    return "\n".join(md)


def main():
    ctx = Contexto()
    org = cf.origenes(ctx.fe)
    hp = cf.hiper_por_origen(ctx.fe, org)
    SALIDA.mkdir(parents=True, exist_ok=True)
    P = {}
    for sem in SEMILLAS:
        carpeta(sem).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        res = Parallel(n_jobs=-1, verbose=2)(delayed(origen)(ctx, i, hp[i], sem) for i in org)
        print(f"semilla {sem}: {time.time() - t0:.0f}s", flush=True)
        P[f"semilla {sem}"] = guardar(ctx, res, f"pronosticos_gtvp_final_h2_sem{sem}.csv", f"GTVP-VAR (semilla {sem})")
    res = Parallel(n_jobs=-1)(delayed(combinado)(ctx, i) for i in org)
    obs = cf.observado()
    prom = pd.concat(P.values()).groupby(["origen", "h", "objetivo"], as_index=False).pronostico.mean()
    prom["error"] = prom.pronostico - prom.objetivo.map(obs)
    P[f"promedio de los pronosticos ({len(SEMILLAS)})"] = prom
    P[f"bosque combinado ({len(SEMILLAS)} x 1000 arboles)"] = guardar(ctx, res, "pronosticos_gtvp_final_h2.csv", "GTVP-VAR")
    txt = tabla(P)
    (cf.RESULTADOS / "02_evaluacion" / "gtvp_final_h2.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
