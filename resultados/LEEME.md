# Resultados de la tesis

Salidas de referencia de la tesis, contra las que `python -m replicacion.correr_todo` compara una replicación. No
contienen el dataset; los CSV de pronósticos incluyen la inflación observada (columna `observado`, dato público del
INDEC).

| Carpeta | Contenido |
|---|---|
| `01_pronosticos/` | pronósticos de todos los modelos: GTVP-VAR (bosque combinado y por semilla), MRF y los de comparación (`lineales_final`: RW, VAR, VAR pequeño, BVAR, FAVAR; `tvp_kk`: TVP-VAR; `tvp_sv`: TVP-VAR-SV; `rf_hiper_gtvp`: RF; `rem`: encuesta REM) |
| `02_evaluacion/` | `evaluacion_final.md` (tablas 2 y 6), `encompassing.md` (tablas 3 y 7), `errores_final.csv`, `gtvp_final_h2.md` (semillas frente al bosque combinado) |
| `03_hiperparametros/` | búsqueda y elección por fecha del GTVP-VAR (`gtvp_var/`), del MRF (`mrf/`) y del RF (`rf/`); `bvar_elegidos.csv` (tabla 8) |
| `04_coeficientes/` | coeficientes del modelo de muestra completa y comparación con el TVP-VAR (figura 3) |
| `05_respuestas/` | respuestas generalizadas puntuales y bandas por simulación anidada (figura 4, tabla 4) |
| `06_traspaso/` | traspaso a 12 meses de una devaluación en cada mes, 2004-2026 (figura 5) |
| `07_descomposicion_varianza/` | GFEVD mensual a 12 meses (figura 6) |
| `08_sensibilidad/` | respuestas con distintos hiperparámetros (figura 7) |
| `figuras_tesis/` | figuras 1 a 7 con el nombre de la tesis (PDF y PNG) |

Formato de los CSV de pronósticos: `modelo, origen, h, objetivo, pronostico, observado, error`, con la inflación en
variación simple mensual (0,02 = 2%).
