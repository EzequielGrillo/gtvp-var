"""Utilidades para usar GTVPVAR como un VAR(p): armado de las matrices con p rezagos y pronostico iterado."""
import numpy as np


def matrices_var(y, rezagos=1, constante=True):
    """Matrices de un VAR(p) a partir de la serie y (T x K).

    Devuelve X ((T - p) x k), con x_t = [1, y_{t-1}, ..., y_{t-p}] (k = 1 + K p, o K p sin constante), e
    Y ((T - p) x K), con la fila t de Y igual a y_t. La fila i de X e Y corresponde al periodo i + p de y, de modo
    que el estado que acompana a cada fila es S[p:].
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    T, K = y.shape
    p = int(rezagos)
    if p < 1 or p >= T:
        raise ValueError(f"rezagos debe estar entre 1 y {T - 1}")
    X = np.column_stack([y[p - j:T - j] for j in range(1, p + 1)])
    if constante:
        X = np.column_stack([np.ones(T - p), X])
    return X, y[p:]


def pronosticar(modelo, y, S, h, rezagos=1, constante=True):
    """Pronostico iterado a 1..h pasos desde el final de y.

    modelo : GTVPVAR ajustado con matrices_var(..., rezagos, constante).
    y      : historia (T x K); se usan sus ultimos `rezagos` periodos.
    S      : estado para cada paso. Un vector (p_s,) mantiene el estado fijo; una matriz (h x p_s) da el estado de cada
             paso; una funcion f(paso, y_ext) -> vector lo calcula con la historia extendida por los pronosticos.
    Devuelve (h x K).
    """
    y_ext = np.array(y, dtype=np.float64)
    if y_ext.ndim == 1:
        y_ext = y_ext[:, None]
    K, p = y_ext.shape[1], int(rezagos)
    out = np.empty((h, K))
    for paso in range(h):
        if callable(S):
            s = np.asarray(S(paso, y_ext), dtype=np.float64)
        else:
            S_arr = np.asarray(S, dtype=np.float64)
            s = S_arr if S_arr.ndim == 1 else S_arr[paso]
        x = np.concatenate([y_ext[-j] for j in range(1, p + 1)])
        if constante:
            x = np.r_[1.0, x]
        B = modelo.predict_beta(s[None, :])[0]                  # K x k
        out[paso] = B @ x
        y_ext = np.vstack([y_ext, out[paso]])
    return out
