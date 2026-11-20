# Encompassing de pronósticos: GTVP-VAR y TVP-VAR

Escenario dic-23 fuera de la estimación; GTVP-VAR con la regla (1000 árboles), TVP-VAR con prior de Koop y Korobilis. Objetivos 2017-01 a 2026-08 sin dic-23 (115 por h). H0: A encompassa a B (B no agrega información); H1 de una cola: B agrega. λ: peso óptimo de B en la combinación (1 − λ)·A + λ·B. No anidados: Harvey, Leybourne y Newbold (1998). Anidados: Clark y West (2007). * 10%, ** 5%, *** 1%.

## ¿TVP-VAR agrega información a GTVP-VAR?

| | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 |
|---|---:|---:|---:|---:|---:|---:|
| λ (peso de TVP-VAR) | 0.84 | 0.69 | 0.12 | -0.02 | -0.34 | -0.39 |
| p no anidados (HLN) | 0.139 | 0.073* | 0.309 | 0.524 | 0.756 | 0.765 |
| p anidados (Clark-West) | 0.137 | 0.069* | 0.304 | 0.525 | 0.765 | 0.777 |
| λ con pérdida absoluta (LAD) | 1.24 | 0.45 | 0.36 | -0.13 | -0.33 | 0.01 |
| p con pérdida absoluta | 0.049** | 0.052* | 0.084* | 0.563 | 0.586 | 0.483 |

## ¿GTVP-VAR agrega información a TVP-VAR?

| | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 |
|---|---:|---:|---:|---:|---:|---:|
| λ (peso de GTVP-VAR) | 0.16 | 0.31 | 0.88 | 1.02 | 1.34 | 1.39 |
| p no anidados (HLN) | 0.310 | 0.185 | 0.051* | 0.096* | 0.096* | 0.101 |
| p anidados (Clark-West) | 0.308 | 0.181 | 0.046** | 0.088* | 0.086* | 0.089* |
| λ con pérdida absoluta (LAD) | -0.24 | 0.55 | 0.64 | 1.13 | 1.33 | 0.99 |
| p con pérdida absoluta | 0.641 | 0.026** | 0.028** | 0.029** | 0.043** | 0.055* |
