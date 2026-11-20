"""Demostracion con datos simulados: GTVP-VAR frente a otros modelos.

Proceso generador (K = 3, VAR(2)), con coeficientes que dependen de un estado observable:

    y_t = c(r_t) + A1(r_t, z_t) y_{t-1} + A2 y_{t-2} + u_t,

- r_t: regimen persistente (cadena de Markov 0/1); cambia la persistencia (0,15 o 0,7 en la diagonal de A1) y la
  constante;
- z_t: variable continua AR(1); el efecto de y2 sobre y1 es 0,4 tanh(2 z_t);
- A2 = 0,1 I, fija.
El estado es S_t = [r_t, z_t, 4 columnas de ruido, tendencia], conocido en t; para pronosticar desde t todos los
modelos usan S_t (el GTVP-VAR se estima con y_{t+1} = B(S_t) x_{t+1} + u_{t+1}).

Pronostico pseudo fuera de muestra de las ultimas 150 observaciones a h = 1 y h = 3, reestimando cada 25 periodos
con ventana creciente; se repite con 5 simulaciones (semillas 0 a 4). Modelos:
- RW: paseo aleatorio;
- VAR(2) por MCO, con ventana creciente y con ventana movil de 80 periodos (aproximacion simple a un TVP-VAR);
- RF: bosque aleatorio de scikit-learn, directo por horizonte, con [y_t, y_{t-1}, S_t] como regresores;
- MRF: GTVPVAR con una ecuacion por variable, directo por horizonte;
- GTVP-VAR(2): iterado, con el estado fijo en su valor del origen (S_o);
- Verdadero: los coeficientes del proceso (solo h = 1, como cota).
Salidas: ejemplos/resultados_simulacion/{simulacion.md, coeficientes.png}
Uso: python ejemplos/simulacion.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

from gtvpvar import GTVPVAR, matrices_var, pronosticar  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "resultados_simulacion"
T, K, P, N_OOS, REFIT, VENTANA_MOVIL, N_REP = 600, 3, 2, 150, 25, 80, 5
HORIZONTES = (1, 3)
# Hiperparametros elegidos con otra simulacion (semilla 123), no con la de la evaluacion. Sin empuje de la tendencia:
# en este proceso los coeficientes no cambian con el tiempo.
PARAMS = dict(ridge_lambda=0.5, rw_regularization=0.5, mtry=0.8, min_node_size=10, block_size=6, n_jobs=-1)


def coeficientes(r, z):
    """c (K,), A1 (K x K) y A2 (K x K) del proceso generador."""
    c = np.full(K, 0.3 if r else -0.1)
    A1 = np.diag(np.full(K, 0.7 if r else 0.15))
    A1[0, 1] = 0.4 * np.tanh(2 * z)
    return c, A1, 0.1 * np.eye(K)


def simular(semilla=0):
    rng = np.random.default_rng(semilla)
    n = T + 50
    r = np.zeros(n, dtype=int)
    z = np.zeros(n)
    for t in range(1, n):
        r[t] = r[t - 1] if rng.random() < 0.97 else 1 - r[t - 1]
        z[t] = 0.95 * z[t - 1] + 0.3 * rng.standard_normal()
    y = np.zeros((n, K))
    for t in range(2, n):
        c, A1, A2 = coeficientes(r[t], z[t])
        y[t] = c + A1 @ y[t - 1] + A2 @ y[t - 2] + 0.3 * rng.standard_normal(K)
    S = np.column_stack([r, z, rng.standard_normal((n, 4)), np.arange(n)])
    return y[50:], S[50:], r[50:], z[50:]


def var_mco(X, Y):
    return np.linalg.lstsq(X, Y, rcond=None)[0].T                 # K x k


def iterar_lineal(B, y_hist, h):
    y_ext = y_hist.copy()
    for _ in range(h):
        x = np.r_[1.0, *[y_ext[-j] for j in range(1, P + 1)]]
        y_ext = np.vstack([y_ext, B @ x])
    return y_ext[-1]


def una_simulacion(semilla):
    """RMSE relativo al VAR por MCO de cada modelo y horizonte, en una simulacion."""
    y, S, r, z = simular(semilla)
    X, Y = matrices_var(y, rezagos=P)                              # fila i <-> periodo i + P
    S_var = S[P - 1:-1]                                            # estado conocido al pronosticar: S_{t-1} para y_t
    origenes = range(T - N_OOS - max(HORIZONTES), T - 1)           # origen o: pronostica o + h
    pron = {h: {m: {} for m in ("RW", "VAR", "VAR movil", "RF", "MRF", "GTVP-VAR", "Verdadero")} for h in HORIZONTES}
    modelos = {}
    for o in origenes:
        if (o - origenes[0]) % REFIT == 0:                         # reestimacion con datos hasta o
            n = o - P + 1                                          # filas de X/Y con periodo <= o
            modelos["VAR"] = var_mco(X[:n], Y[:n])
            modelos["VAR movil"] = var_mco(X[n - VENTANA_MOVIL:n], Y[:n][-VENTANA_MOVIL:])
            modelos["GTVP-VAR"] = GTVPVAR(n_estimators=300, random_state=[1, o],
                                          **PARAMS).fit(X[:n], S_var[:n], Y[:n])
            Z = np.column_stack([y[1:], y[:-1], S[1:]])            # regresores en t (fila t - 1): y_t, y_{t-1}, S_t
            for h in HORIZONTES:
                tr = np.arange(0, o - h)                           # filas con objetivo t + h <= o
                modelos[("RF", h)] = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=o,
                                                           n_jobs=-1).fit(Z[tr], y[tr + 1 + h])
                modelos[("MRF", h)] = [GTVPVAR(n_estimators=200, random_state=[2, o, h, q],
                                               **PARAMS).fit(np.column_stack([np.ones(len(tr)), Z[tr, :2 * K]]),
                                                             S[tr + 1], y[tr + 1 + h, q]) for q in range(K)]
        for h in HORIZONTES:
            if not T - N_OOS - h <= o < T - h:                     # N_OOS objetivos por horizonte
                continue
            pron[h]["RW"][o] = y[o]
            pron[h]["VAR"][o] = iterar_lineal(modelos["VAR"], y[:o + 1], h)
            pron[h]["VAR movil"][o] = iterar_lineal(modelos["VAR movil"], y[:o + 1], h)
            pron[h]["GTVP-VAR"][o] = pronosticar(modelos["GTVP-VAR"], y[:o + 1], S[o], h, rezagos=P)[-1]
            z_o = np.r_[y[o], y[o - 1], S[o]]
            pron[h]["RF"][o] = modelos[("RF", h)].predict(z_o[None])[0]
            x_o = np.r_[1.0, y[o], y[o - 1]]
            pron[h]["MRF"][o] = np.array([m.predict_beta(S[o][None])[0, 0] @ x_o for m in modelos[("MRF", h)]])
            if h == 1:
                c, A1, A2 = coeficientes(r[o + 1], z[o + 1])
                pron[h]["Verdadero"][o] = c + A1 @ y[o] + A2 @ y[o - 1]
    return relativos(pron, y)


def relativos(pron, y):
    """RMSE relativo al VAR(2) por MCO, promedio de las K variables."""
    rmse = {}
    for h in HORIZONTES:
        obj = sorted(pron[h]["VAR"])
        real = np.array([y[o + h] for o in obj])
        for m, p in pron[h].items():
            if p:
                e = np.array([p[o] for o in obj]) - real
                rmse[m, h] = np.sqrt((e ** 2).mean(0))
    return {(m, h): float((v / rmse["VAR", h]).mean()) for (m, h), v in rmse.items()}


def main():
    res = []
    for semilla in range(N_REP):
        res.append(una_simulacion(semilla))
        print(f"simulacion {semilla + 1}/{N_REP}: GTVP-VAR h=1 {res[-1]['GTVP-VAR', 1]:.3f}", flush=True)
    tabla(res)
    figura(*simular(0))


def tabla(res):
    filas = ["| Modelo | " + " | ".join(f"h={h}" for h in HORIZONTES) + " |", "|---|" + "---:|" * len(HORIZONTES)]
    for m in ("RW", "VAR", "VAR movil", "RF", "MRF", "GTVP-VAR", "Verdadero"):
        celdas = []
        for h in HORIZONTES:
            v = np.array([r[m, h] for r in res if (m, h) in r])
            celdas.append(f"{v.mean():.3f} ({v.std(ddof=1):.3f})" if len(v) else "—")
        if m == "GTVP-VAR":
            celdas = [f"**{c}**" for c in celdas]
            m = "**GTVP-VAR**"
        filas.append(f"| {m} | " + " | ".join(celdas) + " |")
    txt = "\n".join(["# Demostracion con datos simulados", "",
                     f"RMSE relativo al VAR({P}) por MCO (promedio de las {K} variables), {N_OOS} pronosticos fuera de "
                     f"muestra por horizonte; media y desvio (entre parentesis) en {N_REP} simulaciones. Menor es mejor.",
                     "", *filas, ""])
    SALIDA.mkdir(exist_ok=True)
    (SALIDA / "simulacion.md").write_text(txt, encoding="utf-8")
    print(txt)


def figura(y, S, r, z):
    """Coeficientes de la muestra completa frente a los verdaderos."""
    X, Y = matrices_var(y, rezagos=P)
    m = GTVPVAR(n_estimators=500, random_state=3, **PARAMS).fit(X, S[P:], Y)
    B = m.predict_beta(S[P:])
    verdad = np.array([coeficientes(r[t], z[t])[1] for t in range(P, T)])
    movil = np.full((T - P, K, X.shape[1]), np.nan)
    for i in range(VENTANA_MOVIL, T - P):
        movil[i] = var_mco(X[i - VENTANA_MOVIL:i], Y[i - VENTANA_MOVIL:i])
    t = np.arange(P, T)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), sharex=True)
    for a, (i, j, titulo) in zip(ax, [(0, 0, "Persistencia de $y_1$ (depende del régimen)"),
                                      (0, 1, "Efecto de $y_2$ sobre $y_1$ (depende de $z_t$)")]):
        a.plot(t, verdad[:, i, j], color="black", lw=1.2, label="Verdadero")
        a.plot(t, B[:, i, 1 + j], color="#0072B2", lw=1.0, label="GTVP-VAR")
        a.plot(t, movil[:, i, 1 + j], color="#E69F00", lw=0.9, ls="--", label=f"VAR móvil ({VENTANA_MOVIL})")
        a.set_title(titulo, fontsize=10)
        a.set_xlabel("t")
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(SALIDA / "coeficientes.png", dpi=150)


if __name__ == "__main__":
    main()
