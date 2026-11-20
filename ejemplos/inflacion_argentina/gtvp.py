"""Datos del GTVP-VAR y ajuste del bosque en un origen."""
import numpy as np

from gtvpvar import GTVPVAR

from . import config as cf
from .estado import Estado


class Contexto:
    """Base, estado S, variables del VAR (Z), parte lineal X = [1, Z] y reglas de avance del estado."""

    def __init__(self):
        d = cf.base()
        self.fe = d.index
        self.S_cols = [k for k in d.columns if k != "Y"]
        self.S, self.Z = d[self.S_cols].values, d[cf.VARS].values
        self.X = np.column_stack([np.ones(len(d)), self.Z])
        self.estado = Estado(self.S_cols)
        self.F = self.fe.get_loc(cf.FUERA)

    def ventana(self, i, sin_dic23=True):
        """Filas de entrenamiento del bosque que pronostica desde i: los ultimos VENT pares (t, t + 1) con t + 1 <= i,
        sin el par cuyo objetivo es dic-2023."""
        tr = np.arange(max(0, i - cf.VENT), i)
        return tr[tr + 1 != self.F] if sin_dic23 else tr

    def ajustar(self, i, semilla, **hp):
        """Bosque con datos hasta el origen i; semilla: random_state."""
        tr = self.ventana(i)
        return GTVPVAR(random_state=semilla, **cf.parametros_bosque(self.S_cols, **hp)).fit(
            self.X[tr], self.S[tr], self.Z[tr + 1])

    @staticmethod
    def beta_media(*bosques):
        """beta(s): promedio de las matrices de todos los arboles de los bosques."""
        return lambda s: np.concatenate([m.betas_arboles(s)[:, 0] for m in bosques]).mean(0)


def hp_dict(hp):
    """(lambda, zeta, mtry, bloque, nodo) -> argumentos del bosque."""
    lam, zeta, mtry, blq, nodo = hp
    return dict(ridge_lambda=lam, rw_regularization=zeta, mtry=mtry, block_size=blq, min_node_size=nodo)
