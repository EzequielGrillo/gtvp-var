"""GTVP-VAR: VAR con parametros que varian segun el estado, estimados por un bosque aleatorio (Grillo, 2026).
El Macroeconomic Random Forest (Goulet Coulombe, 2024) es el caso de una sola ecuacion."""
from .modelo import GTVPVAR

__all__ = ["GTVPVAR"]
