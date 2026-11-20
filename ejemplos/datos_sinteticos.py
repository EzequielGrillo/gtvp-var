"""Ejemplo minimo del GTVP-VAR con datos sinteticos.

Se simula un VAR(1) bivariado cuya persistencia cambia segun un regimen observable:

    y_t = c + A(s_t) y_{t-1} + u_t,   A = 0,2 I si s_t <= 0 y A = 0,8 I si s_t > 0,

y se estima con GTVPVAR usando como estado S = [s_t, ruido, ruido, tendencia]. El bosque deberia recuperar las dos
matrices A segun el valor de s_t.

Uso: python ejemplos/datos_sinteticos.py
"""
import numpy as np

from gtvpvar import GTVPVAR


def simular(T=400, semilla=0):
    rng = np.random.default_rng(semilla)
    s = np.sign(np.sin(np.arange(T + 1) / 25.0)) + 0.1 * rng.standard_normal(T + 1)     # regimen persistente
    y = np.zeros((T + 1, 2))
    for t in range(1, T + 1):
        A = 0.8 if s[t] > 0 else 0.2
        y[t] = 0.1 + A * y[t - 1] + 0.2 * rng.standard_normal(2)
    X = np.column_stack([np.ones(T), y[:-1]])                       # x_t = [1, y_{t-1}]
    S = np.column_stack([s[1:], rng.standard_normal((T, 2)), np.arange(T)])
    return X, S, y[1:]


def main(n_estimators=200, n_jobs=1):
    X, S, Y = simular()
    m = GTVPVAR(n_estimators=n_estimators, ridge_lambda=0.15, rw_regularization=0.3, mtry=0.5, min_node_size=30,
                block_size=12, trend_push_column=S.shape[1] - 1, n_jobs=n_jobs, random_state=1)
    m.fit(X, S, Y)
    regimenes = np.array([[-1.0, 0.0, 0.0, 200.0], [1.0, 0.0, 0.0, 200.0]])
    B = m.predict_beta(regimenes)                                   # (2 regimenes, K = 2, k = 3)
    for nombre, b in zip(("s <= 0 (A = 0,2 I)", "s > 0  (A = 0,8 I)"), B):
        print(f"Regimen {nombre}: diagonal de A estimada = {np.round(np.diag(b[:, 1:]), 2)}")
    return B


if __name__ == "__main__":
    main(n_jobs=-1)
