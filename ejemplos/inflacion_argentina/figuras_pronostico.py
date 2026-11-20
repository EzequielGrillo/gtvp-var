"""Figuras de los pronosticos.
- pronosticos: inflacion observada y pronosticada a 1, 3 y 6 meses (escala logaritmica; los pronosticos por debajo de
  0,8% quedan en el piso del eje).
- cumsum_mae: diferencia acumulada de error absoluto respecto del paseo aleatorio (y la version cuadratica), por fecha
  del objetivo; dic-2023 aporta 0 y se sombrea. Lineas verticales: cambios de gobierno (dic-2019, dic-2023).
Salidas: resultados/figuras/{pronosticos, cumsum_mae, cumsum_rmse}.{pdf,png}
Uso: python -m ejemplos.inflacion_argentina.figuras_pronostico
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from . import config as cf  # noqa: E402
from .estilo import ESTILO, PAPER  # noqa: E402
from .evaluacion import errores  # noqa: E402

SALIDA = cf.RESULTADOS / "figuras"
MARCA = {"GTVP-VAR": "o", "TVP-VAR": "s", "VAR": "^", "BVAR": "D", "FAVAR": "v", "RF": "P", "MRF": "X", "RW": "*"}
TRAZO = {"GTVP-VAR": (0, (7, 2)), "TVP-VAR": (0, (3, 1.6)), "VAR": (0, (1, 1.6)), "BVAR": (0, (6, 1.6, 1.4, 1.6)),
         "FAVAR": (0, (9, 2, 1.4, 2)), "RF": (0, (2.4, 1.2)), "MRF": (0, (4, 1.2, 1.2, 1.2, 1.2, 1.2)),
         "RW": (0, (5, 1.5, 1, 1.5, 1, 1.5, 1, 1.5))}
CORTES = ["2019-12-01", "2023-12-01"]


def figura_pronosticos(E):
    modelos = ["GTVP-VAR", "TVP-VAR", "VAR", "BVAR", "FAVAR", "RF", "MRF", "RW"]
    obs = cf.observado()
    plt.rcParams.update(PAPER)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.9), sharex=True, sharey=True)
    for ax, h in zip(axs, [1, 3, 6]):
        o = E["GTVP-VAR"].loc[h].sort_index()
        y = 100 * pd.Series(o.index.map(obs), index=o.index)
        ax.axvspan(cf.FUERA - pd.Timedelta(days=15), cf.FUERA + pd.Timedelta(days=15), color="#e6e6e6", lw=0, zorder=0)
        for m in modelos[::-1]:
            p = (100 * (E[m].loc[h].sort_index() + pd.Series(o.index.map(obs), index=o.index))).clip(lower=0.8)
            ax.plot(p.index, p.values, color=ESTILO[m][0], ls=TRAZO[m], marker=MARCA[m], ms=3.6,
                    markevery=(modelos.index(m) % 6, 6), markerfacecolor="white" if m != "GTVP-VAR" else "#6B6B6B",
                    markeredgewidth=0.8, lw=1.2, zorder=5 if m == "GTVP-VAR" else 3)
        ax.plot(y.index, y.values, color="black", lw=1.6, alpha=1, ls="-", zorder=4, solid_capstyle="round")
        ax.set_title(f"$h = {h}$", loc="left")
        ax.axhline(0, color="black", lw=0.5, zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(0.8, 32)
        ax.set_yticks([1, 2, 4, 8, 16, 32])
        ax.set_yticklabels(["1", "2", "4", "8", "16", "32"])
        ax.minorticks_off()
        ax.grid(axis="y", color="#e3e3e3", lw=0.5, zorder=0)
        if h == 1:
            ax.set_ylabel(r"Inflaci$\mathrm{\acute{o}}$n mensual (%)")
        ax.set_xlim(pd.Timestamp("2016-12-01"), pd.Timestamp("2026-10-01"))
    for ax in axs:
        ax.xaxis.set_major_locator(mdates.YearLocator(2, month=1))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelrotation=90)
    hs = [Line2D([], [], color="black", lw=1.6)] + \
         [Line2D([], [], color=ESTILO[m][0], ls=TRAZO[m], lw=1.2, marker=MARCA[m], ms=3.6,
                 markerfacecolor="white" if m != "GTVP-VAR" else "#6B6B6B", markeredgewidth=0.8) for m in modelos]
    fig.tight_layout(rect=(0, 0.19, 1, 1), h_pad=0.6, w_pad=0.5)
    fig.legend(hs, ["Observada"] + modelos, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.005),
               handlelength=2.0, columnspacing=1.0, fontsize=7.5)
    for ext in ("pdf", "png"):
        fig.savefig(SALIDA / f"pronosticos.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _alinear(e, h):
    cal = pd.date_range(cf.INI, cf.FIN, freq="MS")
    x = e.loc[h].reindex(cal)
    mascara = (cal != cf.FUERA) & x.notna().values
    return np.where(mascara, x.fillna(0).values, 0.0), mascara


def _coma(x, _):
    return f"{x:g}".replace(".", ",")


def figura_acumulada(E, cuadratica):
    modelos = ["GTVP-VAR", "TVP-VAR", "VAR", "BVAR", "FAVAR", "RF", "MRF"]
    plt.rcParams.update(PAPER)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), sharex=True)
    cal = pd.date_range(cf.INI, cf.FIN, freq="MS")
    for h, ax in zip([1, 3, 6], axs.ravel()):
        eb, mb = _alinear(E["RW"], h)
        lb = eb ** 2 if cuadratica else np.abs(eb)
        ax.axvspan(cf.FUERA - pd.Timedelta(days=15), cf.FUERA + pd.Timedelta(days=15), color="#e6e6e6", lw=0, zorder=0)
        ax.axhline(0, color="#000000", lw=0.5, zorder=1)
        for m in modelos[::-1]:
            ea, ma = _alinear(E[m], h)
            assert (ma == mb).all(), (m, h)
            la = ea ** 2 if cuadratica else np.abs(ea)
            col, ls, lw = ESTILO[m]
            ax.plot(cal, np.cumsum((la - lb) * (100 ** 2 if cuadratica else 100)), color=col, ls=ls, lw=lw,
                    zorder=5 if m == "GTVP-VAR" else 3)
        ax.set_title(f"$h = {h}$")
        for f in CORTES:
            ax.axvline(pd.Timestamp(f), color="#808080", lw=0.5, ls=(0, (4, 2)), zorder=2)
        bordes = [pd.Timestamp("2017-01-01")] + [pd.Timestamp(f) for f in CORTES] + [pd.Timestamp("2026-09-01")]
        for k, nom in enumerate(["I", "II", "III"]):
            ax.text(bordes[k] + (bordes[k + 1] - bordes[k]) / 2, 0.97, nom, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7, color="#555555", zorder=6,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_coma))
        ax.set_xticks([pd.Timestamp(f"{a}-01-01") for a in (2017, 2020, 2023, 2026)])
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(pd.Timestamp("2016-10-01"), pd.Timestamp("2026-11-01"))
    axs[0].set_ylabel(r"Error cuadr$\mathrm{\acute{a}}$tico acumulado vs. RW (pp$^2$)" if cuadratica
                      else "Error absoluto acumulado vs. RW (pp)")
    hs = [Line2D([], [], color=ESTILO[m][0], ls=ESTILO[m][1], lw=ESTILO[m][2]) for m in modelos]
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.2)
    fig.legend(hs, modelos, ncol=len(modelos), frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               handlelength=2.6, columnspacing=1.4)
    nombre = "cumsum_rmse" if cuadratica else "cumsum_mae"
    for ext in ("pdf", "png"):
        fig.savefig(SALIDA / f"{nombre}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    E = errores()
    figura_pronosticos(E)
    figura_acumulada(E, False)
    figura_acumulada(E, True)


if __name__ == "__main__":
    main()
