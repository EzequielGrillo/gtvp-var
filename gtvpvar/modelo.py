"""GTVP-VAR: VAR con coeficientes que dependen del estado de la economia, estimados por un bosque aleatorio.

    y_t = B(S_t) x_t + u_t,      y_t de K variables, x_t = [1, y_{t-1}, ...] comun a las K ecuaciones

Cada arbol particiona el estado S_t con cortes MULTIVARIADOS: un corte minimiza la suma sobre las K ecuaciones del
objetivo penalizado del Macroeconomic Random Forest (Goulet Coulombe, 2024, ec. 2),

    sum_lado sum_q [ min_b sum_t w(t) (y_qt - x_t'b)^2 + lambda ||b - b0_q||^2 ],

con w(t) el kernel de paseo aleatorio y b0 el MCO de la ventana. Cada hoja guarda un VAR completo (K x k). El MRF
univariado de Goulet Coulombe es el caso K = 1.

Detalles de la implementacion:
- Bootstrap bayesiano por bloques: bloques contiguos de largo aleatorio y pesos Exponencial(1) normalizados
  (Dirichlet(1, ..., 1) de Rubin), un sorteo por arbol.
- Busqueda de cortes con ridge simple (sin kernel de paseo aleatorio); el kernel completo se usa en las hojas.
- Cortes exactos (puntos medios entre valores consecutivos) sobre mtry * p columnas de S sorteadas por nodo; la
  columna de tendencia puede tener una probabilidad de sorteo mayor (trend push).
- X e Y se estandarizan por magnitud (sin centrar) antes del ajuste; los coeficientes se devuelven en las unidades
  originales.
"""
import numpy as np
from joblib import Parallel, delayed

from ._nucleo import barrer, puntos_de_corte, recorrer, ridge_hoja


def _escalas(X, tol=1e-12):
    """Desvio de cada columna (o |media| si es constante, como el intercepto; 1 si es nula)."""
    sd = X.std(axis=0)
    media_abs = np.abs(X.mean(axis=0))
    return np.ascontiguousarray(np.where(sd > tol, sd, np.where(media_abs > tol, media_abs, 1.0)), dtype=np.float64)


def _pesos_bootstrap(T, bloque, rng):
    """Bootstrap bayesiano por bloques: T etiquetas en floor(T / bloque) grupos ordenados (bloques contiguos de largo
    aleatorio), peso Exponencial(1) por bloque, normalizado a media 1."""
    n_bloques = max(1, int(T // bloque))
    grupos = np.sort(rng.integers(0, n_bloques, size=T))
    w = rng.exponential(1.0, size=n_bloques)[grupos]
    return w / w.mean()


class _Buffers:
    def __init__(self, T, k, K):
        self.w = np.zeros(T)
        self.tocados = np.empty(T, dtype=np.int64)
        self.obj_izq = np.empty(T)
        self.obj_tot = np.empty(T)
        self.A = np.zeros((k, k))
        self.C = np.zeros((k, K))
        self.wy2 = np.zeros(K)
        self.Atmp = np.zeros((k, k))
        self.Ctmp = np.zeros((k, K))


def _elegir_corte(idx, S, X, Y, B0, cfg, rng, bufs, peso):
    """Mejor (columna, umbral) del nodo, o None si debe ser hoja."""
    if idx.shape[0] < cfg["min_node_size"]:
        return None
    p = S.shape[1]
    m = min(p, max(1, int(round(cfg["mtry"] * p))))
    if cfg["trend_push_column"] is None:
        cols = rng.choice(p, size=m, replace=False)
    else:
        probs = np.full(p, 1.0 / p)
        probs[cfg["trend_push_column"]] *= cfg["trend_push_weight"]
        probs /= probs.sum()
        cols = rng.choice(p, size=m, replace=False, p=probs)
    mejor_col, mejor_obj, mejor_umbral = -1, np.inf, 0.0
    for j in cols:
        vals = S[:, j][idx]
        orden = np.argsort(vals, kind="quicksort")
        val_ord = np.ascontiguousarray(vals[orden])
        idx_ord = np.ascontiguousarray(idx[orden])
        cortes = puntos_de_corte(val_ord)
        if cortes.shape[0] == 0:
            continue
        ci, obj = barrer(idx_ord, val_ord, cortes, X, Y, B0, cfg["lam"], 0.0, Y.shape[0], cfg["min_hijo"], peso,
                         bufs.w, bufs.tocados, bufs.obj_izq, bufs.obj_tot, bufs.A, bufs.C, bufs.wy2,
                         bufs.Atmp, bufs.Ctmp)
        if ci != -1 and obj < mejor_obj:
            mejor_obj, mejor_col, mejor_umbral = obj, int(j), float(cortes[ci])
    if mejor_col == -1:
        return None
    izq = S[:, mejor_col][idx] <= mejor_umbral
    return mejor_col, mejor_umbral, np.ascontiguousarray(idx[izq]), np.ascontiguousarray(idx[~izq])


def _crecer(idx, S, X, Y, B0, cfg, rng, bufs, peso, nodos):
    """Crece el arbol en preorden; nodos: lista de [columna, umbral, izquierdo, derecho, beta de la hoja]."""
    nid = len(nodos)
    nodos.append([-1, 0.0, -1, -1, None])
    corte = _elegir_corte(idx, S, X, Y, B0, cfg, rng, bufs, peso)
    if corte is None:
        nodos[nid][4] = ridge_hoja(idx, X, Y, B0, cfg["lam"], cfg["zeta"], Y.shape[0], bufs.w, bufs.tocados, peso)
        return nid
    col, umbral, izq, der = corte
    nodos[nid][0], nodos[nid][1] = col, umbral
    nodos[nid][2] = _crecer(izq, S, X, Y, B0, cfg, rng, bufs, peso, nodos)
    nodos[nid][3] = _crecer(der, S, X, Y, B0, cfg, rng, bufs, peso, nodos)
    return nid


class Arbol:
    """Arbol compilado: beta de forma (n_nodos, K, k); los nodos internos llevan ceros."""

    def __init__(self, feature, threshold, left, right, leaf_beta):
        self.feature, self.threshold, self.left, self.right, self.leaf_beta = feature, threshold, left, right, leaf_beta

    def hojas(self, S):
        return recorrer(np.ascontiguousarray(S, dtype=np.float64), self.feature, self.threshold, self.left, self.right)

    def predict_beta(self, S):
        return self.leaf_beta[self.hojas(S)]


def _ajustar_arbol(semilla, S, X, Y, B0, cfg):
    """semilla: una semilla (pesos del bootstrap y sorteo de columnas del mismo generador) o un par (semilla de los
    pesos, semilla del sorteo de columnas)."""
    if isinstance(semilla, tuple):
        rng_w, rng = np.random.default_rng(semilla[0]), np.random.default_rng(semilla[1])
    else:
        rng = rng_w = np.random.default_rng(semilla)
    T, K = Y.shape
    k = X.shape[1]
    peso = _pesos_bootstrap(T, cfg["block_size"], rng_w)
    nodos = []
    _crecer(np.arange(T, dtype=np.int64), S, X, Y, B0, cfg, rng, _Buffers(T, k, K), peso, nodos)
    beta = np.zeros((len(nodos), K, k))
    for i, nd in enumerate(nodos):
        if nd[4] is not None:
            beta[i] = nd[4].T
    return Arbol(np.array([nd[0] for nd in nodos], dtype=np.int32), np.array([nd[1] for nd in nodos], dtype=np.float64),
                 np.array([nd[2] for nd in nodos], dtype=np.int32), np.array([nd[3] for nd in nodos], dtype=np.int32),
                 beta)


class GTVPVAR:
    """GTVP-VAR (y MRF con K = 1).

    Parametros
    ----------
    n_estimators : numero de arboles.
    ridge_lambda : penalizacion ridge hacia el MCO de la ventana (lambda).
    rw_regularization : peso del kernel de paseo aleatorio en las hojas (zeta, en [0, 1)).
    mtry : fraccion de columnas de S candidatas en cada nodo.
    min_node_size : tamano minimo de un nodo para intentar cortarlo.
    min_leaf_fraction : cada hijo debe tener al menos ceil(min_leaf_fraction * k) meses.
    block_size : largo medio de los bloques del bootstrap bayesiano.
    trend_push_column, trend_push_weight : columna de S con la tendencia y factor de su probabilidad de sorteo.
    n_jobs : procesos para ajustar los arboles (joblib).
    random_state : semilla (o lista de enteros) de la que salen las semillas de los arboles.
    """

    def __init__(self, n_estimators=1000, ridge_lambda=0.15, rw_regularization=0.9, mtry=0.15, min_node_size=30,
                 min_leaf_fraction=2.0, block_size=12, trend_push_column=None, trend_push_weight=3.0, n_jobs=1,
                 random_state=None):
        self.n_estimators, self.ridge_lambda, self.rw_regularization = n_estimators, ridge_lambda, rw_regularization
        self.mtry, self.min_node_size, self.min_leaf_fraction = mtry, min_node_size, min_leaf_fraction
        self.block_size, self.trend_push_column, self.trend_push_weight = block_size, trend_push_column, trend_push_weight
        self.n_jobs, self.random_state = n_jobs, random_state

    def fit(self, X, S, Y, semillas=None):
        """X (T x k) parte lineal, S (T x p) estado, Y (T x K) variables a explicar (o vector, K = 1).
        semillas: opcional, una semilla (o par) por arbol en lugar de las derivadas de random_state."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        S = np.ascontiguousarray(S, dtype=np.float64)
        Y = np.ascontiguousarray(np.atleast_2d(np.asarray(Y, dtype=np.float64).T).T)
        k = X.shape[1]
        x_esc = _escalas(X)
        y_esc = Y.std(axis=0)
        y_esc[y_esc == 0] = 1.0
        Xf = np.ascontiguousarray(X / x_esc)
        Yf = np.ascontiguousarray(Y / y_esc)
        B0 = np.ascontiguousarray(np.linalg.lstsq(Xf, Yf, rcond=None)[0])          # prior: MCO de la ventana (k x K)
        cfg = dict(lam=self.ridge_lambda, zeta=self.rw_regularization, min_node_size=self.min_node_size,
                   min_hijo=int(np.ceil(self.min_leaf_fraction * k)), mtry=self.mtry, block_size=self.block_size,
                   trend_push_column=self.trend_push_column, trend_push_weight=self.trend_push_weight)
        if semillas is None:
            semillas = np.random.SeedSequence(self.random_state).spawn(self.n_estimators)
        Sf = np.asfortranarray(S)
        if self.n_jobs == 1:
            arboles = [_ajustar_arbol(s, Sf, Xf, Yf, B0, cfg) for s in semillas]
        else:
            arboles = Parallel(n_jobs=self.n_jobs)(delayed(_ajustar_arbol)(s, Sf, Xf, Yf, B0, cfg) for s in semillas)
        esc = y_esc[:, None] / x_esc[None, :]                                        # K x k
        for a in arboles:
            a.leaf_beta *= esc
        self.trees_, self.B0_ = arboles, B0 * esc.T
        self.K_, self.k_ = Y.shape[1], k
        return self

    def betas_arboles(self, S):
        """(n_arboles, n, K, k): matriz del VAR en la hoja de cada arbol para cada fila de S."""
        S = np.atleast_2d(S)
        return np.stack([a.predict_beta(S) for a in self.trees_])

    def predict_beta(self, S):
        """(n, K, k): promedio de las matrices de los arboles."""
        return self.betas_arboles(S).mean(axis=0)

    def guardar(self, ruta):
        n = np.array([len(a.feature) for a in self.trees_], dtype=np.int32)
        np.savez_compressed(ruta, n=n, K=self.K_, k=self.k_,
                            **{c: np.concatenate([getattr(a, c) for a in self.trees_])
                               for c in ("feature", "threshold", "left", "right", "leaf_beta")})

    @classmethod
    def cargar(cls, ruta):
        z = np.load(ruta)
        cortes = np.r_[0, np.cumsum(z["n"])]
        m = cls.__new__(cls)
        m.K_, m.k_ = int(z["K"]), int(z["k"])
        arr = {c: z[c] for c in ("feature", "threshold", "left", "right", "leaf_beta")}
        m.trees_ = [Arbol(*(np.ascontiguousarray(arr[c][cortes[b]:cortes[b + 1]])
                            for c in ("feature", "threshold", "left", "right", "leaf_beta")))
                    for b in range(len(z["n"]))]
        return m
