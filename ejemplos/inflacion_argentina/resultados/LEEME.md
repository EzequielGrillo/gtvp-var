# Resultados de referencia

Salidas del ejemplo, contra las que `python -m ejemplos.inflacion_argentina.correr_todo` compara una corrida nueva. No
contienen el dataset; los CSV de pronósticos incluyen la inflación observada (columna `observado`, dato público del
INDEC).

| Carpeta | Contenido |
|---|---|
| `01_pronosticos/` | pronósticos de todos los modelos: GTVP-VAR (bosque combinado y por semilla), MRF y los de comparación (`lineales_final`: RW, VAR, VAR pequeño, BVAR, FAVAR; `tvp_kk`: TVP-VAR; `tvp_sv`: TVP-VAR-SV; `rf_hiper_gtvp`: RF; `rem`: encuesta REM) |
| `02_evaluacion/` | `evaluacion_final.md` (MAE y RMSE relativos al RW), `encompassing.md` (encompassing), `errores_final.csv`, `gtvp_final_h2.md` (semillas frente al bosque combinado) |
| `03_hiperparametros/` | búsqueda y elección por fecha del GTVP-VAR (`gtvp_var/`), del MRF (`mrf/`) y del RF (`rf/`); `bvar_elegidos.csv` (BVAR) |
| `04_coeficientes/` | coeficientes del modelo de muestra completa y comparación con el TVP-VAR |
| `05_respuestas/` | respuestas generalizadas puntuales y bandas por simulación anidada |
| `06_traspaso/` | traspaso a 12 meses de una devaluación en cada mes, 2004-2026 |
| `07_descomposicion_varianza/` | GFEVD mensual a 12 meses |
| `08_sensibilidad/` | respuestas con distintos hiperparámetros |
| `figuras/` | figuras de los pronósticos, el error acumulado y el análisis estructural (PDF y PNG) |

Formato de los CSV de pronósticos: `modelo, origen, h, objetivo, pronostico, observado, error`, con la inflación en
variación simple mensual (0,02 = 2%).
