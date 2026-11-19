# GTVP-VAR, especificación final

Estado de comun.base_gtvp() (ESTADO_GTVP.md), regla de hiperparámetros, dic-23 fuera de la estimación, 1000 árboles por semilla. MAE / RW (p-valor GW, pérdida absoluta), objetivos 2017-01 a 2026-08 sin dic-23; observado = expm1(IPC_t) de Dataset_base.xlsx.

| | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 |
|---|---:|---:|---:|---:|---:|---:|
| semilla 1 | 0.875* (0.084) | 0.776* (0.056) | 0.724* (0.070) | 0.655* (0.061) | 0.649* (0.065) | 0.684* (0.089) |
| semilla 2 | 0.872* (0.076) | 0.778* (0.057) | 0.726* (0.073) | 0.660* (0.066) | 0.652* (0.067) | 0.692* (0.099) |
| semilla 3 | 0.879* (0.095) | 0.778* (0.054) | 0.724* (0.069) | 0.653* (0.062) | 0.652* (0.068) | 0.687* (0.095) |
| promedio de los pronósticos (3) | 0.875* (0.084) | 0.777* (0.055) | 0.724* (0.070) | 0.656* (0.063) | 0.651* (0.066) | 0.687* (0.093) |
| bosque combinado (3 x 1000 árboles) | 0.875* (0.084) | 0.777* (0.055) | 0.724* (0.070) | 0.655* (0.063) | 0.650* (0.066) | 0.686* (0.093) |
| (final con hiperparámetros anteriores, 3 semillas) | 0.878* (0.094) | 0.779* (0.058) | 0.732* (0.077) | 0.661* (0.068) | 0.652* (0.073) | 0.691 (0.106) |
| (original: estado de 73 col., semilla 1) | 0.884 (0.112) | 0.790* (0.066) | 0.735* (0.079) | 0.664* (0.071) | 0.659* (0.076) | 0.693 (0.102) |