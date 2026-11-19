"""Definiciones comunes de la replicacion: rutas, datos, muestra de evaluacion y especificacion del GTVP-VAR."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos"                          # Dataset_base.xlsx (no se distribuye; ver README)
RESULTADOS = Path(os.environ.get("GTVP_RESULTADOS", RAIZ / "resultados"))   # otra carpeta: pruebas
TRABAJO = Path(os.environ.get("GTVP_TRABAJO", RAIZ / "trabajo"))                    # bosques guardados y archivos intermedios (se puede borrar)

# Muestra de evaluacion: objetivos 2017-01 a 2026-08, h = 1..6; dic-2023 queda fuera de la estimacion y de las metricas
INI, FIN = pd.Timestamp("2017-01-01"), pd.Timestamp("2026-08-01")
FUERA = pd.Timestamp("2023-12-01")
H = 6
VENT = 272                                      # pares de la ventana del GTVP-VAR (todos los disponibles en 2016-07)
ARBOLES = int(os.environ.get("GTVP_ARBOLES", 1000))   # GTVP_ARBOLES: corridas de prueba reducidas

# Parte lineal del VAR y bloques del estado S segun como avanza cada serie al iterar
VARS = ["IPC_t", "TC_t", "Brecha_t", "Tasa_t", "BrechaPIB_t-3"]
ESTADO_VAR = ["IPC", "TC", "Brecha", "Tasa", "BrechaPIB"]           # su pronostico
ESTADO_IDENTIDAD = ["CCL", "TCR", "BC", "RRII"]                     # identidades contables
ESTADO_EXOGENAS = ["T10Y", "IPMP", "vIPMP", "Deficit"]              # fijas en el promedio de los 12 meses al origen
ESTADO_SATELITE = ["M2", "BM", "M2PIB", "X", "Q", "Sal", "MERVAL", "varRIN", "IPI", "CredReal"]   # BVARX(1) satelite
ESTADO_FUERA = ["Desempleo", "CompraRIN"]                           # en la base, fuera del estado

# Hiperparametros: fechas de ajuste de la regla anual (julio de cada ano) y grilla
AJUSTES = [pd.Timestamp(f"{y}-07-01") for y in range(2016, 2026)]


def base():
    """Y + las 70 columnas del estado S (23 series x 3 versiones + tendencia t, ultima)."""
    d = pd.read_excel(DATOS / "Dataset_base.xlsx")
    d = d.set_index(pd.to_datetime(d.pop("Fecha")))
    assert (d["Y"] == d["IPC_t"]).all()
    d = d.drop(columns=[k for k in d.columns if k.split("_t")[0] in ESTADO_FUERA])
    series = {k.split("_t")[0] for k in d.columns if k not in ("Y", "t")}
    assert series == set(ESTADO_VAR + ESTADO_IDENTIDAD + ESTADO_EXOGENAS + ESTADO_SATELITE), series
    assert d.shape[1] == 71 and d.columns[-1] == "t"
    return d


def observado():
    """Inflacion mensual observada (variacion simple): exp(IPC_t) - 1, con IPC_t la log-variacion de la base."""
    d = pd.read_excel(DATOS / "Dataset_base.xlsx", usecols=["Fecha", "IPC_t"])
    return pd.Series(np.expm1(d["IPC_t"].values), index=pd.to_datetime(d["Fecha"]), name="IPC_t")


def origenes(fe):
    """Origenes de los pronosticos iterados: 2016-07 a 2026-07 (cada uno da h = 1..6)."""
    return list(range(fe.get_loc(INI) - H, fe.get_loc(FIN)))


def parametros_bosque(S_cols, **cambios):
    """Configuracion del bosque (cambios: ridge_lambda, rw_regularization, mtry, block_size, min_node_size, ...)."""
    p = dict(n_estimators=ARBOLES, ridge_lambda=0.15, rw_regularization=0.30, mtry=0.15, min_node_size=30,
             min_leaf_fraction=2.0, block_size=12, trend_push_column=S_cols.index("t"), trend_push_weight=3.0)
    p.update(cambios)
    return p


def eleccion(T):
    """Hiperparametros elegidos en la fecha de ajuste T (busqueda del GTVP-VAR)."""
    return pd.read_csv(RESULTADOS / "03_hiperparametros" / "gtvp_var" / f"eleccion_T{T:%Y-%m}_sindic23_v36_vm18.csv").iloc[0]


def hiper_por_origen(fe, org):
    """(lambda, zeta, mtry, bloque, nodo minimo) por origen: la eleccion de julio previa a cada origen."""
    E = {T: eleccion(T) for T in AJUSTES}
    out = {}
    for i in org:
        e = E[max(t for t in AJUSTES if t <= fe[i])]
        out[i] = (float(e.lam), float(e.zeta), float(e.mtry), int(e.block), int(e.nodo))
    return out
