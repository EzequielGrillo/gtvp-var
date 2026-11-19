"""Pruebas de capacidad predictiva: Giacomini y White (2006) y encompassing de Harvey, Leybourne y Newbold (1998).
El objetivo excluido (dic-2023) se trata como un hueco en la serie de diferencias (serie modulada en amplitud)."""
import numpy as np
from scipy import stats


def gw(ea, eb, h, mascara, absoluta=True):
    """p-valor del test de Giacomini-White de igual capacidad predictiva (incondicional, HAC Bartlett con h-1 rezagos)."""
    d = (np.abs(ea) - np.abs(eb)) if absoluta else (ea ** 2 - eb ** 2)
    n, mu = mascara.sum(), d[mascara].mean()
    u = np.where(mascara, d - mu, 0.0)
    om = u @ u / n
    for L in range(1, h):
        om += 2 * (1 - L / h) * (u[L:] @ u[:-L]) / n
    return stats.chi2.sf(n * mu ** 2 / om, 1)


def estrellas(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
