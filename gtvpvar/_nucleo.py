"""Rutinas compiladas (numba) del GTVP-VAR: acumuladores del kernel de paseo aleatorio, ridge hacia el prior,
barrido de cortes y estimacion de las hojas."""
import numpy as np
from numba import njit

_JITTER = 1e-12          # piso relativo a la traza: evita matrices singulares sin alterar problemas bien condicionados


@njit(cache=True, inline="always")
def estabilizar(A):
    k = A.shape[0]
    tr = 0.0
    for a in range(k):
        tr += A[a, a]
    eps = _JITTER * (tr / k if tr > 0.0 else 1.0)
    for a in range(k):
        A[a, a] += eps


@njit(cache=True)
def puntos_de_corte(valores_ordenados):
    """Puntos medios entre valores distintos consecutivos (cortes exactos)."""
    if valores_ordenados.shape[0] < 2 or valores_ordenados[0] == valores_ordenados[-1]:
        return np.empty(0, dtype=np.float64)
    uniq = np.unique(valores_ordenados)
    return (uniq[:-1] + uniq[1:]) / 2.0


@njit(cache=True)
def pesos_podio(idx, zeta, T, w, tocados):
    """Kernel de paseo aleatorio (Goulet Coulombe, 2024, sec. 2.3): peso 1 a los meses de la hoja, zeta a los vecinos
    a +-1 y zeta^2 a +-2 (el maximo si se superponen). Escribe w (en cero al entrar), registra en `tocados` las
    posiciones escritas y devuelve cuantas son."""
    n = idx.shape[0]
    n_t = 0
    for i in range(n):
        t = idx[i]
        if w[t] == 0.0:
            tocados[n_t] = t
            n_t += 1
        w[t] = 1.0
    zeta2 = zeta * zeta
    for i in range(n):
        t = idx[i]
        for dt in (-1, 1):
            tt = t + dt
            if 0 <= tt < T and w[tt] < zeta:
                if w[tt] == 0.0:
                    tocados[n_t] = tt
                    n_t += 1
                w[tt] = zeta
    for i in range(n):
        t = idx[i]
        for dt in (-2, 2):
            tt = t + dt
            if 0 <= tt < T and w[tt] < zeta2:
                if w[tt] == 0.0:
                    tocados[n_t] = tt
                    n_t += 1
                w[tt] = zeta2
    return n_t


@njit(cache=True)
def limpiar(w, tocados, n_t):
    for i in range(n_t):
        w[tocados[i]] = 0.0


@njit(cache=True, inline="always")
def _agregar(s, zeta, T, w, tocados, n_t, A, C, wy2, X, Y, peso):
    """Suma el mes s (y sus vecinos del kernel) a X'WX, X'WY e y'Wy de un lado del corte."""
    k = X.shape[1]
    K = Y.shape[1]
    for d in range(-2, 3):
        t = s + d
        if t < 0 or t >= T:
            continue
        ad = d if d >= 0 else -d
        if ad == 0:
            cand = 1.0
        elif ad == 1:
            cand = zeta
        else:
            cand = zeta * zeta
        prev = w[t]
        if cand > prev:
            if prev == 0.0:
                tocados[n_t] = t
                n_t += 1
            delta = (cand - prev) * peso[t]
            w[t] = cand
            for q in range(K):
                yq = Y[t, q]
                wy2[q] += delta * yq * yq
            for a in range(k):
                xa = delta * X[t, a]
                for q in range(K):
                    C[a, q] += xa * Y[t, q]
                for b in range(k):
                    A[a, b] += xa * X[t, b]
    return n_t


@njit(cache=True)
def _objetivo(A, C, lam, B0, wy2, Atmp, Ctmp):
    """Suma sobre las K ecuaciones de min_b sum_t w_t (y_qt - x_t'b)^2 + lam ||b - b0_q||^2."""
    k = A.shape[0]
    K = C.shape[1]
    const = np.zeros(K)
    for a in range(k):
        for q in range(K):
            Ctmp[a, q] = C[a, q] + lam * B0[a, q]
            const[q] += lam * B0[a, q] * B0[a, q]
        for b in range(k):
            Atmp[a, b] = A[a, b]
        Atmp[a, a] += lam
    estabilizar(Atmp)
    beta = np.linalg.solve(Atmp, Ctmp)
    tot = 0.0
    for q in range(K):
        dot = 0.0
        for a in range(k):
            dot += beta[a, q] * Ctmp[a, q]
        tot += wy2[q] - dot + const[q]
    return tot


@njit(cache=True)
def barrer(idx_ord, val_ord, cortes, X, Y, B0, lam, zeta, T, min_hijo, peso, w, tocados, obj_izq, obj_tot,
           A, C, wy2, Atmp, Ctmp):
    """Recorre los cortes de una columna (de izquierda a derecha y de derecha a izquierda, acumulando) y devuelve el
    indice del mejor corte y su objetivo (-1, inf si ninguno deja al menos min_hijo meses de cada lado)."""
    n = idx_ord.shape[0]
    ncut = cortes.shape[0]
    k = X.shape[1]
    K = Y.shape[1]
    for sentido in range(2):
        for a in range(k):
            for q in range(K):
                C[a, q] = 0.0
            for b in range(k):
                A[a, b] = 0.0
        for q in range(K):
            wy2[q] = 0.0
        n_t = 0
        cnt = 0
        if sentido == 0:
            ptr = 0
            for ci in range(ncut):
                corte = cortes[ci]
                while ptr < n and val_ord[ptr] <= corte:
                    n_t = _agregar(idx_ord[ptr], zeta, T, w, tocados, n_t, A, C, wy2, X, Y, peso)
                    ptr += 1
                    cnt += 1
                if cnt < min_hijo or n - cnt < min_hijo:
                    obj_izq[ci] = np.inf
                else:
                    obj_izq[ci] = _objetivo(A, C, lam, B0, wy2, Atmp, Ctmp)
        else:
            ptr = n - 1
            for ci in range(ncut - 1, -1, -1):
                corte = cortes[ci]
                while ptr >= 0 and val_ord[ptr] > corte:
                    n_t = _agregar(idx_ord[ptr], zeta, T, w, tocados, n_t, A, C, wy2, X, Y, peso)
                    ptr -= 1
                    cnt += 1
                if obj_izq[ci] == np.inf:
                    obj_tot[ci] = np.inf
                    continue
                der = _objetivo(A, C, lam, B0, wy2, Atmp, Ctmp)
                obj_tot[ci] = np.inf if der == np.inf else obj_izq[ci] + der
        for i in range(n_t):
            w[tocados[i]] = 0.0
    mejor, mejor_obj = -1, np.inf
    for ci in range(ncut):
        if obj_tot[ci] < mejor_obj:
            mejor_obj = obj_tot[ci]
            mejor = ci
    return mejor, mejor_obj


@njit(cache=True)
def ridge_hoja(idx, X, Y, B0, lam, zeta, T, w, tocados, peso):
    """Coeficientes de la hoja (k x K): ridge hacia B0 con el kernel de paseo aleatorio completo."""
    n_t = pesos_podio(idx, zeta, T, w, tocados)
    k = X.shape[1]
    K = Y.shape[1]
    A = np.zeros((k, k))
    C = np.zeros((k, K))
    for ii in range(n_t):
        t = tocados[ii]
        wt = w[t] * peso[t]
        for a in range(k):
            xa = X[t, a]
            for q in range(K):
                C[a, q] += wt * xa * Y[t, q]
            for b in range(k):
                A[a, b] += wt * xa * X[t, b]
    limpiar(w, tocados, n_t)
    for a in range(k):
        A[a, a] += lam
        for q in range(K):
            C[a, q] += lam * B0[a, q]
    estabilizar(A)
    return np.linalg.solve(A, C)


@njit(cache=True)
def recorrer(S, feature, threshold, left, right):
    """Hoja de destino de cada fila de S (convencion S <= umbral -> izquierda, la misma que en el ajuste)."""
    n = S.shape[0]
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        nodo = 0
        while feature[nodo] != -1:
            if S[i, feature[nodo]] <= threshold[nodo]:
                nodo = left[nodo]
            else:
                nodo = right[nodo]
        out[i] = nodo
    return out
