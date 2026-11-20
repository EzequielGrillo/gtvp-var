"""Proyeccion del estado S al iterar el GTVP-VAR (h > 1).

En cada paso, con el pronostico z del VAR:
- variables del VAR: se corren sus rezagos y la version mas reciente toma el pronostico; la tendencia suma 1;
- identidades: CCL = Brecha_t - Brecha_t-1 + TC_t; TCR_t = TCR_t-1 + TC_t - IPC_t; BC = exp(X) - exp(Q);
  RRII_t = RRII_t-1 + varRIN_t;
- exogenas (T10Y, IPMP, vIPMP, Deficit): fijas en su promedio de los 12 meses previos al origen;
- satelites: un BVARX(1) Minnesota (hiperparametros por verosimilitud marginal) condicionado a las variables del VAR,
  estimado con la ventana que termina en el origen.
"""
import re

import numpy as np
from scipy.special import multigammaln

from . import config as cf

KAPPA_X = 10.0                    # escala del prior de los regresores exogenos del BVARX


def grupos_rezagos(S_cols):
    """serie -> indices de sus columnas, de la version mas reciente a la mas vieja."""
    g = {}
    for i, c in enumerate(S_cols):
        m = re.fullmatch(r"(.+)_t(?:-(\d+))?", c)
        if m:
            g.setdefault(m.group(1), []).append((int(m.group(2) or 0), i))
    return {b: [i for _, i in sorted(v)] for b, v in g.items()}


def _logml_nw(Ys, Xs):
    T, n = Ys.shape
    k = Xs.shape[1]
    XX = Xs.T @ Xs
    B = np.linalg.solve(XX, Xs.T @ Ys)
    E = Ys - Xs @ B
    nu = T - k
    return (-n * nu / 2 * np.log(np.pi) + multigammaln(nu / 2, n) - n / 2 * np.linalg.slogdet(XX)[1]
            - nu / 2 * np.linalg.slogdet(E.T @ E)[1]), B


def bvarx(Yb, W):
    """BVARX(1): Yb_t = c + A Yb_t-1 + D0 W_t + D1 W_t-1, prior Minnesota por observaciones ficticias con (lambda,
    delta) elegidos por verosimilitud marginal. Devuelve (B, lambda, delta)."""
    T1, n = Yb.shape
    m = W.shape[1]
    sig = np.empty(n)
    for i in range(n):
        X1 = np.column_stack([np.ones(T1 - 1), Yb[:-1, i]])
        e = Yb[1:, i] - X1 @ np.linalg.lstsq(X1, Yb[1:, i], rcond=None)[0]
        sig[i] = np.sqrt(e @ e / (len(e) - 2))
    sw = W.std(0) + 1e-12
    X = np.column_stack([np.ones(T1 - 1), Yb[:-1], W[1:], W[:-1]])
    y = Yb[1:]
    k = X.shape[1]
    best = (-np.inf, None, None, None)
    for lam in np.geomspace(0.01, 10.0, 31):
        for dl in np.round(np.arange(0.0, 1.01, 0.1), 2):
            Yd = np.vstack([np.diag(dl * sig) / lam, np.zeros((2 * m, n)), np.zeros((1, n)), np.diag(sig)])
            Xd = np.zeros((n + 2 * m + 1 + n, k))
            Xd[:n, 1:1 + n] = np.diag(sig) / lam
            Xd[n:n + 2 * m, 1 + n:] = np.diag(np.r_[sw, sw]) / (lam * KAPPA_X)
            Xd[n + 2 * m, 0] = 1e-3
            l_all, B = _logml_nw(np.vstack([Yd, y]), np.vstack([Xd, X]))
            l_d, _ = _logml_nw(Yd, Xd)
            if l_all - l_d > best[0]:
                best = (l_all - l_d, B.T, lam, dl)
    return best[1], best[2], best[3]


class Estado:
    """Indices de las columnas de S y reglas de avance."""

    def __init__(self, S_cols):
        self.g = g = grupos_rezagos(S_cols)
        self.i_t = S_cols.index("t")
        self.var = [(g[b], j) for j, b in enumerate(cf.ESTADO_VAR)]
        self.cz = [g[b][0] for b in cf.ESTADO_VAR]
        self.sat = [g[b] for b in cf.ESTADO_SATELITE]
        ix = {n: S_cols.index(n) for n in ["Brecha_t", "Brecha_t-1", "TC_t", "TCR_t-1", "IPC_t"]}
        self.i_brecha, self.i_brecha1, self.i_tc, self.i_tcr1, self.i_ipc = (
            ix["Brecha_t"], ix["Brecha_t-1"], ix["TC_t"], ix["TCR_t-1"], ix["IPC_t"])

    def origen(self, S, i):
        """Lo que se fija en el origen i: promedios de las exogenas y BVARX satelite."""
        ext = [(self.g[b], float(S[i - 11:i + 1, self.g[b][0]].mean())) for b in cf.ESTADO_EXOGENAS]
        w = S[max(0, i - cf.VENT):i + 1]
        Bm, _, _ = bvarx(w[:, [c[0] for c in self.sat]], w[:, self.cz])
        return ext, Bm

    def avanzar(self, viejo, z, ctx):
        """Estado del paso siguiente, dado el estado actual y el pronostico z del VAR."""
        ext, Bm = ctx
        g = self.g
        s = viejo.copy()
        for cols, j in self.var:
            s[cols[1:]] = viejo[cols[:-1]]
            s[cols[0]] = z[j]
        s[self.i_t] = viejo[self.i_t] + 1
        for b in ("CCL", "TCR"):
            s[g[b][1:]] = viejo[g[b][:-1]]
        s[g["CCL"][0]] = s[self.i_brecha] - s[self.i_brecha1] + s[self.i_tc]
        s[g["TCR"][0]] = s[self.i_tcr1] + s[self.i_tc] - s[self.i_ipc]
        for cols, v in ext:
            s[cols] = v
        nuevo = Bm @ np.r_[1.0, viejo[[c[0] for c in self.sat]], s[self.cz], viejo[self.cz]]
        for cols, v in zip(self.sat, nuevo):
            s[cols[1:]] = viejo[cols[:-1]]
            s[cols[0]] = v
        s[g["BC"][1:]] = viejo[g["BC"][:-1]]
        s[g["BC"][0]] = np.exp(s[g["X"][0]]) - np.exp(s[g["Q"][0]])
        s[g["RRII"][1:]] = viejo[g["RRII"][:-1]]
        s[g["RRII"][0]] = viejo[g["RRII"][0]] + s[g["varRIN"][0]]
        return s

    def iterar(self, beta, i, S, Z, ctx=None):
        """Pronostico iterado h = 1..H desde el origen i; beta(s) -> matriz del VAR (K x (1 + K)).
        Devuelve [(i, h, z_h)]."""
        ctx = self.origen(S, i) if ctx is None else ctx
        z, s, out = Z[i].copy(), S[i].copy(), []
        for h in range(1, cf.H + 1):
            z = beta(s) @ np.r_[1.0, z]
            out.append((i, h, z.copy()))
            s = self.avanzar(s, z, ctx)
        return out
