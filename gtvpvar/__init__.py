"""GTVP-VAR: VAR con parametros que varian segun el estado, estimados por un bosque aleatorio.
El Macroeconomic Random Forest (Goulet Coulombe, 2024) es el caso de una sola ecuacion."""
from .modelo import GTVPVAR
from .var import matrices_var, pronosticar

__all__ = ["GTVPVAR", "matrices_var", "pronosticar"]
