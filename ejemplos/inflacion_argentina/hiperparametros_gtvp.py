"""Busqueda anual de hiperparametros del GTVP-VAR (validacion pseudo fuera de muestra).

En cada fecha de ajuste T (julio, 2016-2026), solo con datos hasta T:
- grilla lambda {0,15; 0,5; 1} x zeta {0,3; 0,6; 0,9} x mtry {0,075; 0,15} x bloque {12; 6} x nodo minimo {30; 15};
- validacion: los 36 objetivos hasta T; el bosque (200 arboles) se reestima cada 6 meses con la ventana que termina en
  la reestimacion y pronostica iterando desde cada origen de esos 6 meses (objetivos <= T);
- criterio: promedio sobre h = 1..6 del MAE del IPC relativo al del paseo aleatorio, con pesos 0,5^(edad / 18) por
  antiguedad del objetivo; dic-2023 fuera del entrenamiento y de los errores;
- eleccion: la mejor; entre las que quedan a menos de 0,01 de ella, la mas regularizada (lambda mas alto, mtry mas
  bajo, zeta mas alto, nodo mas grande, bloque mas largo).
Salida: resultados/03_hiperparametros/gtvp_var/{busqueda,eleccion}_T{T}_sindic23_v36_vm18.csv
Uso: python -m ejemplos.inflacion_argentina.hiperparametros_gtvp [T=2026-07 ...]   (sin argumentos: las 11 fechas)
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as cf
from .gtvp import Contexto, hp_dict

GRILLA = list(itertools.product([0.15, 0.5, 1.0], [0.3, 0.6, 0.9], [0.075, 0.15], [12, 6], [30, 15]))
VAL_MESES, REFIT, VIDA_MEDIA, TOL = 36, 6, 18.0, 0.01
ARBOLES = int(os.environ.get("GTVP_ARBOLES_BUSQUEDA", 200))
SALIDA = cf.RESULTADOS / "03_hiperparametros" / "gtvp_var"


def tarea(ctx, r, combo, origenes, i_T, CTXS):
    m = ctx.ajustar(r, [1, 700, r, 3], n_estimators=ARBOLES, **hp_dict(combo))
    beta = ctx.beta_media(m)
    return [(combo, o, h, z) for o in origenes
            for _, h, z in ctx.estado.iterar(beta, o, ctx.S, ctx.Z, CTXS[o]) if o + h <= i_T]


def buscar(ctx, T):
    fe, Z = ctx.fe, ctx.Z
    i_T = fe.get_loc(T)
    i_v = fe.get_loc(T - pd.DateOffset(months=VAL_MESES - 1))
    origenes = list(range(i_v - 1, i_T))                                 # objetivos o + h en [T - 35, T]
    t0 = time.time()
    CTXS = dict(zip(origenes, Parallel(n_jobs=-1)(delayed(ctx.estado.origen)(ctx.S, o) for o in origenes)))
    tareas = [(r, cb, [o for o in origenes if r <= o < r + REFIT]) for r in origenes[::REFIT] for cb in GRILLA]
    res = Parallel(n_jobs=-1, verbose=2)(delayed(tarea)(ctx, r, cb, oo, i_T, CTXS) for r, cb, oo in tareas)
    print(f"{T:%Y-%m}: {len(tareas)} bosques en {time.time() - t0:.0f}s", flush=True)
    reg = [dict(combo=combo, o=o, h=h, e_ipc=z[0] - Z[o + h, 0], e_rw=Z[o, 0] - Z[o + h, 0],
                e_sis=(z - Z[o + h]) if h == 1 else None)
           for r in res for combo, o, h, z in r if o + h != ctx.F]
    R = pd.DataFrame(reg)
    var_sis = Z[[o + 1 for o in origenes if o + 1 != ctx.F]].var(0)
    tabla = []
    for combo, g in R.groupby("combo"):
        w = 0.5 ** ((i_T - (g.o + g.h)) / VIDA_MEDIA)
        uno = pd.Series(1.0, index=g.index)

        def mae_rel(hh, pesos):
            k = g.h == hh
            return (pesos[k] * np.abs(g.e_ipc[k])).sum() / (pesos[k] * np.abs(g.e_rw[k])).sum()
        mae_h = [mae_rel(h, w) for h in range(1, 7)]
        g1 = g[g.h == 1]
        tabla.append(dict(lam=combo[0], zeta=combo[1], mtry=combo[2], block=combo[3], nodo=combo[4],
                          principal=np.mean(mae_h), principal_sin_pesos=np.mean([mae_rel(h, uno) for h in range(1, 7)]),
                          ipc_h1=mae_h[0], sistema_h1=float(np.mean([np.sum(e ** 2 / var_sis) for e in g1.e_sis])),
                          **{f"mae_h{h}": v for h, v in zip(range(1, 7), mae_h)}))
    Tb = pd.DataFrame(tabla).sort_values("principal").reset_index(drop=True)
    cerca = Tb[Tb.principal <= Tb.principal.min() + TOL]
    eleg = cerca.sort_values(["lam", "mtry", "zeta", "nodo", "block"], ascending=[False, True, False, False, False]).iloc[0]
    tag = f"{T:%Y-%m}_sindic23_v{VAL_MESES}_vm{VIDA_MEDIA:g}"
    SALIDA.mkdir(parents=True, exist_ok=True)
    Tb.to_csv(SALIDA / f"busqueda_T{tag}.csv", index=False)
    pd.DataFrame([dict(T=f"{T:%Y-%m}", lam=eleg.lam, zeta=eleg.zeta, mtry=eleg.mtry, block=eleg.block, nodo=eleg.nodo,
                       criterio=eleg.principal, n_cerca=len(cerca), mejor=Tb.principal.min())]).to_csv(
        SALIDA / f"eleccion_T{tag}.csv", index=False)
    print(f"  elegida: lambda {eleg.lam:g}, zeta {eleg.zeta:g}, mtry {eleg.mtry:g}, bloque {eleg.block:g}, "
          f"nodo {eleg.nodo:g} (criterio {eleg.principal:.3f})", flush=True)


def main():
    fechas = [pd.Timestamp(a[2:] + "-01") for a in sys.argv[1:] if a.startswith("T=")] or \
             [pd.Timestamp(f"{y}-07-01") for y in range(2016, 2027)]
    ctx = Contexto()
    for T in fechas:
        buscar(ctx, T)


if __name__ == "__main__":
    main()
