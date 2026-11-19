"""Corre la replicacion de la tesis de punta a punta, en orden y de forma reanudable.

Etapas (entre parentesis, el tiempo aproximado con 12 nucleos):
  1. hiperparametros  busqueda anual de hiperparametros del GTVP-VAR, Anexo D             (~3 h)
  2. pronosticos      pronosticos pseudo fuera de muestra del GTVP-VAR, 3 semillas        (~1 h)
  3. mrf              busqueda y pronosticos del MRF                                       (~6 h)
  4. evaluacion       tablas 2, 3, 6 y 7                                                   (~1 min)
  5. figuras          figuras 1 y 2                                                        (~1 min)

Cada etapa se salta si sus salidas ya existen (usar --rehacer para forzarla). Los resultados se escriben en una
carpeta nueva (por defecto replica/), que se inicializa con los pronosticos de los modelos de comparacion (RW, VAR,
BVAR, FAVAR, TVP-VAR, TVP-VAR-SV, RF y REM), que no se recalculan. Al final se comparan las salidas con resultados/.

Uso:
  python -m replicacion.correr_todo                       # todas las etapas
  python -m replicacion.correr_todo evaluacion figuras    # solo algunas
  python -m replicacion.correr_todo --salida otra/ --rehacer
  python -m replicacion.correr_todo --verificar           # solo compara replica/ con resultados/
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
REFERENCIA = RAIZ / "resultados"
COMPARACION = ["pronosticos_lineales_final.csv", "pronosticos_tvp_kk.csv", "pronosticos_tvp_sv.csv",
               "pronosticos_rf_hiper_gtvp.csv", "pronosticos_rem.csv"]

# etapa -> (modulo, argumentos, salidas que indican que esta hecha)
ETAPAS = {
    "hiperparametros": ("hiperparametros_gtvp", [],
                        [f"03_hiperparametros/gtvp_var/eleccion_T{a}-07_sindic23_v36_vm18.csv" for a in range(2016, 2027)]),
    "pronosticos": ("pronosticos_gtvp", [],
                    ["01_pronosticos/pronosticos_gtvp_final_h2.csv", "02_evaluacion/gtvp_final_h2.md"]),
    "mrf": ("mrf", [], ["01_pronosticos/pronosticos_mrf_hiper.csv"]),
    "evaluacion": ("evaluacion", [], ["02_evaluacion/evaluacion_final.md", "02_evaluacion/encompassing.md",
                                      "02_evaluacion/errores_final.csv"]),
    "figuras": ("figuras_pronostico", [], ["figuras_tesis/fig1_pronosticos.png", "figuras_tesis/fig2_cumsum_mae.png",
                                           "figuras_tesis/cumsum_rmse.png"]),
}


def preparar(salida):
    """Crea la carpeta de salida con los pronosticos de los modelos de comparacion."""
    dest = salida / "01_pronosticos"
    dest.mkdir(parents=True, exist_ok=True)
    for a in COMPARACION:
        if not (dest / a).exists():
            shutil.copy2(REFERENCIA / "01_pronosticos" / a, dest / a)


def correr(nombre, salida, rehacer):
    modulo, args, salidas = ETAPAS[nombre]
    if not rehacer and all((salida / s).exists() for s in salidas):
        print(f"[{nombre}] ya hecha, se salta (--rehacer para forzarla)", flush=True)
        return
    print(f"[{nombre}] python -m replicacion.{modulo} {' '.join(args)}", flush=True)
    t0 = time.time()
    entorno = dict(os.environ, GTVP_RESULTADOS=str(salida))
    subprocess.run([sys.executable, "-m", f"replicacion.{modulo}", *args], cwd=RAIZ, env=entorno, check=True)
    print(f"[{nombre}] lista en {(time.time() - t0) / 60:.1f} min", flush=True)


def verificar(salida):
    """Compara las salidas de la replica con resultados/: texto byte a byte; CSV tambien numericamente."""
    import numpy as np
    import pandas as pd

    print(f"\nComparacion de {salida} contra {REFERENCIA}:")
    for _, _, salidas in ETAPAS.values():
        for s in salidas:
            a, b = REFERENCIA / s, salida / s
            if not b.exists():
                print(f"  falta     {s}")
            elif s.endswith(".png"):
                print(f"  figura    {s} (comparar a ojo o pixel a pixel; depende de las fuentes instaladas)")
            elif a.read_bytes() == b.read_bytes():
                print(f"  identico  {s}")
            elif s.endswith(".csv"):
                x, y = (pd.read_csv(p) for p in (a, b))
                num = x.select_dtypes("number").columns
                dif = float(np.nanmax(np.abs(x[num].values - y[num].values))) if x.shape == y.shape else np.inf
                print(f"  distinto  {s} (max |dif| = {dif:.2e})")
            else:
                print(f"  distinto  {s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapas", nargs="*", metavar="etapa", help=f"etapas a correr: {', '.join(ETAPAS)} (por defecto, todas)")
    ap.add_argument("--salida", type=Path, default=RAIZ / "replica", help="carpeta de resultados (por defecto replica/)")
    ap.add_argument("--rehacer", action="store_true", help="corre las etapas aunque sus salidas existan")
    ap.add_argument("--verificar", action="store_true", help="solo compara la carpeta de salida con resultados/")
    a = ap.parse_args()
    if set(a.etapas) - set(ETAPAS):
        ap.error(f"etapas desconocidas: {', '.join(sorted(set(a.etapas) - set(ETAPAS)))}")
    salida = a.salida.resolve()
    if salida == REFERENCIA.resolve():
        sys.exit("La salida no puede ser resultados/: es la referencia contra la que se compara.")
    if not a.verificar:
        if not (RAIZ / "datos" / "Dataset_base.xlsx").exists():
            sys.exit("Falta datos/Dataset_base.xlsx (no se distribuye; ver datos/LEEME.md).")
        preparar(salida)
        for e in a.etapas or list(ETAPAS):
            correr(e, salida, a.rehacer)
    verificar(salida)


if __name__ == "__main__":
    main()
