"""MRF de Goulet Coulombe (2024): el GTVP-VAR con una sola ecuacion, directo por horizonte,

    pi_{t+h} = x_t' beta_h(S_t),   x_t = [1, IPC_t, TC_t, Brecha_t, Tasa_t, BrechaPIB_t-3],

con el mismo estado S que el GTVP-VAR. Ventana fija por h: los pares (s, s + h) disponibles en el primer origen de ese
h (279 - 2h), sin el par con objetivo dic-2023.

Busqueda de hiperparametros (como la del GTVP-VAR, adaptada a un modelo directo): en cada julio de 2016 a 2025,
lambda {0,15; 0,5; 1} x zeta {0,3; 0,6; 0,9} x mtry {0,075; 0,15} x bloque {12; 6} con el nodo minimo que eligio el
GTVP-VAR en esa fecha; 36 objetivos de validacion con pesos 0,5^(edad / 18), reestimacion cada 6 meses, 200 arboles;
criterio: promedio sobre h del MAE / RW; desempate por la combinacion mas regularizada. Una combinacion para todos los h.
Salidas: resultados/03_hiperparametros/mrf/{busqueda,eleccion}_T{T}.csv,
         resultados/01_pronosticos/pronosticos_mrf_hiper.csv
Uso: python -m ejemplos.inflacion_argentina.mrf [buscar] [pronosticar]   (sin argumentos: las dos etapas)
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from gtvpvar import GTVPVAR

from . import config as cf

GRILLA = list(itertools.product([0.15, 0.5, 1.0], [0.3, 0.6, 0.9], [0.075, 0.15], [12, 6]))
VAL_MESES, VIDA_MEDIA, REFIT, TOL = 36, 18.0, 6, 0.01
ARBOLES_VAL = int(os.environ.get("GTVP_ARBOLES_BUSQUEDA", 200))
SALIDA_HP = cf.RESULTADOS / "03_hiperparametros" / "mrf"


class Datos:
    def __init__(self):
        d = cf.base()
        self.fe = d.index
        self.S_cols = [k for k in d.columns if k != "Y"]
        self.S = d[self.S_cols].values
        self.X = np.column_stack([np.ones(len(d)), d[cf.VARS].values])
        self.y = d["IPC_t"].values
        self.F, i0 = self.fe.get_loc(cf.FUERA), self.fe.get_loc(cf.INI)
        self.vent = {h: i0 - 2 * h + 1 for h in range(1, cf.H + 1)}

    def ajustar(self, r, h, semilla, n_estimators, hp):
        """Bosque directo a h con los ultimos vent[h] pares (s, s + h), s + h <= r."""
        lam, zeta, mtry, blq, nodo = hp
        tr = np.arange(max(0, r - h + 1 - self.vent[h]), r - h + 1)
        tr = tr[tr + h != self.F]
        p = cf.parametros_bosque(self.S_cols, n_estimators=n_estimators, ridge_lambda=lam, rw_regularization=zeta,
                                 mtry=mtry, block_size=blq, min_node_size=nodo)
        return GTVPVAR(random_state=semilla, **p).fit(self.X[tr], self.S[tr], self.y[tr + h][:, None])


def _validar(D, r, h, combo, origenes):
    m = D.ajustar(r, h, [1, 700, r, h], ARBOLES_VAL, combo)
    B = m.predict_beta(D.S[origenes])[:, 0]
    return [(combo, o, h, float(b @ D.X[o])) for o, b in zip(origenes, B)]


def buscar(D):
    fe, y, F = D.fe, D.y, D.F
    SALIDA_HP.mkdir(parents=True, exist_ok=True)
    for T in cf.AJUSTES:
        t0 = time.time()
        nodo = int(cf.eleccion(T).nodo)
        grilla = [(lam, zeta, mtry, blq, nodo) for lam, zeta, mtry, blq in GRILLA]
        i_T = fe.get_loc(T)
        i_v = i_T - VAL_MESES + 1
        tareas = []
        for r in range(i_v - cf.H, i_T, REFIT):
            for h in range(1, cf.H + 1):
                oo = [o for o in range(r, r + REFIT) if i_v <= o + h <= i_T and o + h != F]
                if oo:
                    tareas += [(r, h, cb, oo) for cb in grilla]
        res = Parallel(n_jobs=-1, verbose=2)(delayed(_validar)(D, r, h, cb, oo) for r, h, cb, oo in tareas)
        R = pd.DataFrame([dict(lam=cb[0], zeta=cb[1], mtry=cb[2], block=cb[3], nodo=cb[4], o=o, h=h, e=p - y[o + h],
                               e_rw=y[o] - y[o + h], w=0.5 ** ((i_T - (o + h)) / VIDA_MEDIA))
                          for rr in res for cb, o, h, p in rr])
        tab = []
        for k, g in R.groupby(["lam", "zeta", "mtry", "block", "nodo"]):
            mae_h = [(g.w[g.h == h] * g.e[g.h == h].abs()).sum() / (g.w[g.h == h] * g.e_rw[g.h == h].abs()).sum()
                     for h in range(1, cf.H + 1)]
            tab.append(dict(lam=k[0], zeta=k[1], mtry=k[2], block=k[3], nodo=k[4], principal=np.mean(mae_h),
                            **{f"mae_h{h}": v for h, v in zip(range(1, cf.H + 1), mae_h)}))
        Tb = pd.DataFrame(tab).sort_values("principal").reset_index(drop=True)
        cerca = Tb[Tb.principal <= Tb.principal.min() + TOL]
        el = cerca.sort_values(["lam", "mtry", "zeta", "nodo", "block"], ascending=[False, True, False, False, False]).iloc[0]
        Tb.to_csv(SALIDA_HP / f"busqueda_T{T:%Y-%m}.csv", index=False)
        pd.DataFrame([dict(T=f"{T:%Y-%m}", lam=el.lam, zeta=el.zeta, mtry=el.mtry, block=int(el.block), nodo=int(el.nodo),
                           criterio=el.principal, n_cerca=len(cerca), mejor=Tb.principal.min())]).to_csv(
            SALIDA_HP / f"eleccion_T{T:%Y-%m}.csv", index=False)
        print(f"{T:%Y-%m}: lambda {el.lam:g}, zeta {el.zeta:g}, mtry {el.mtry:g}, bloque {int(el.block)}, nodo "
              f"{int(el.nodo)} (criterio {el.principal:.3f}) [{time.time() - t0:.0f}s]", flush=True)


def _pronosticar(D, i, h, hp):
    m = D.ajustar(i, h, [1, 500, i, h], cf.ARBOLES, hp)
    return i, h, float(m.predict_beta(D.S[i:i + 1])[0, 0] @ D.X[i])


def pronosticar(D):
    fe = D.fe
    i0 = fe.get_loc(cf.INI)
    tareas = [(i, h) for h in range(1, cf.H + 1) for i in range(i0 - h, fe.get_loc(cf.FIN) - h + 1)]
    E = {T: pd.read_csv(SALIDA_HP / f"eleccion_T{T:%Y-%m}.csv").iloc[0] for T in cf.AJUSTES}
    hp = {}
    for i in sorted({i for i, _ in tareas}):
        e = E[max([t for t in cf.AJUSTES if t <= fe[i]], default=cf.AJUSTES[0])]
        hp[i] = (float(e.lam), float(e.zeta), float(e.mtry), int(e.block), int(e.nodo))
    t0 = time.time()
    res = Parallel(n_jobs=-1, verbose=5)(delayed(_pronosticar)(D, i, h, hp[i]) for i, h in tareas)
    print(f"MRF: {len(tareas)} bosques en {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame([dict(modelo="MRF", origen=fe[i], h=h, objetivo=fe[i + h], pronostico=np.expm1(p))
                       for i, h, p in res])
    df["observado"] = df.objetivo.map(cf.observado())
    df["error"] = df.pronostico - df.observado
    df.to_csv(cf.RESULTADOS / "01_pronosticos" / "pronosticos_mrf_hiper.csv", index=False)


def main():
    etapas = [a for a in sys.argv[1:] if a in ("buscar", "pronosticar")] or ["buscar", "pronosticar"]
    D = Datos()
    if "buscar" in etapas:
        buscar(D)
    if "pronosticar" in etapas:
        pronosticar(D)


if __name__ == "__main__":
    main()
