"""Estilo de las figuras: tipografia Computer Modern (como LaTeX) y paleta Okabe-Ito (apta para daltonismo)."""
PAPER = {"font.family": ["cmr10", "STIXGeneral"], "mathtext.fontset": "cm", "axes.unicode_minus": False,
         "axes.formatter.use_mathtext": True, "font.size": 8, "axes.linewidth": 0.6, "xtick.direction": "in",
         "ytick.direction": "in", "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3,
         "ytick.major.size": 3, "xtick.top": False, "ytick.right": False, "axes.titlesize": 9, "legend.fontsize": 8,
         "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "white",
         "axes.edgecolor": "black", "axes.spines.top": True, "axes.spines.right": True, "xtick.color": "black",
         "ytick.color": "black", "axes.labelcolor": "black", "axes.grid": False}

# modelo -> (color, estilo de linea, grosor)
ESTILO = {"GTVP-VAR": ("#6B6B6B", "-", 1.3), "TVP-VAR": ("#0072B2", (0, (5, 1.5)), 0.8), "VAR": ("#009E73", "-", 0.7),
          "BVAR": ("#E69F00", (0, (6, 1.5, 1.5, 1.5)), 0.8), "FAVAR": ("#CC79A7", (0, (2, 1)), 0.8),
          "RF": ("#D55E00", (0, (4, 1, 1, 1, 1, 1)), 0.8), "MRF": ("#56B4E9", (0, (1, 1)), 1.0),
          "RW": ("#B3A100", "-", 1.2)}
