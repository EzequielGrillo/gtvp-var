# Ejemplo: pronóstico de inflación en Argentina

Aplicación completa del GTVP-VAR: pronóstico pseudo fuera de muestra de la inflación mensual de Argentina, con
selección de hiperparámetros en tiempo real y comparación contra modelos estándar.

## Especificación

- **VAR (K = 5):** inflación, variación del tipo de cambio oficial, brecha cambiaria, tasa de interés y brecha del
  producto (con 3 meses de rezago). `x_t = [1, y_{t-1}]`.
- **Estado S (70 columnas):** 23 series macroeconómicas con su rezago de publicación y dos rezagos más (monetarias,
  externas, fiscales, financieras y de actividad), más una tendencia. Detalle en [`datos/LEEME.md`](datos/LEEME.md).
- **Pronósticos iterados a h = 1..6:** al iterar, el estado se proyecta con identidades contables, promedios de 12
  meses para las exógenas y un BVARX(1) satélite para el resto (`estado.py`).
- **Muestra:** orígenes de 2016-07 a 2026-07; ventana móvil de 272 meses; diciembre de 2023 (devaluación del 118%)
  queda fuera de la estimación y de las métricas.
- **Hiperparámetros:** cada julio se elige la combinación de la grilla (λ × ζ × mtry × bloque × nodo mínimo) con mejor
  MAE relativo al paseo aleatorio en los 36 meses previos, ponderando más los meses recientes; entre las que quedan a
  menos de 0,01, la más regularizada.
- **Pronóstico final:** bosque de 3 × 1000 árboles (tres semillas combinadas).
- **MRF:** el mismo modelo con una sola ecuación, directo por horizonte.

## Cómo correrlo

Con el dataset en `datos/` y las versiones de `requirements.txt`, desde la raíz del repositorio:

```bash
python -m ejemplos.inflacion_argentina.correr_todo
```

| Etapa | Script | Tiempo (12 núcleos) | Salida |
|---|---|---|---|
| hiperparametros | `hiperparametros_gtvp.py` | ~3 h | `resultados/03_hiperparametros/gtvp_var/` |
| pronosticos | `pronosticos_gtvp.py` | ~1 h | `resultados/01_pronosticos/pronosticos_gtvp_final_h2*.csv` |
| mrf | `mrf.py` | ~6 h | `resultados/03_hiperparametros/mrf/`, `pronosticos_mrf_hiper.csv` |
| evaluacion | `evaluacion.py` | ~1 min | `resultados/02_evaluacion/` |
| figuras | `figuras_pronostico.py` | ~1 min | `resultados/figuras/{pronosticos,cumsum_mae,cumsum_rmse}` |

El orquestador escribe en `replica/`, salta las etapas ya hechas y al final compara contra `resultados/`. Cada script
también se puede correr solo, por ejemplo `python -m ejemplos.inflacion_argentina.evaluacion`.

Variables de entorno: `GTVP_RESULTADOS` (carpeta de salida), `GTVP_TRABAJO` (bosques guardados, por defecto
`trabajo/`), `GTVP_ARBOLES` y `GTVP_ARBOLES_BUSQUEDA` (árboles del pronóstico y de las búsquedas, para corridas
reducidas de prueba).

Los modelos de comparación (RW, VAR, VAR pequeño, BVAR, FAVAR, TVP-VAR, TVP-VAR-SV, RF y la encuesta REM) son
implementaciones estándar: no se incluyen, pero sus pronósticos están en `resultados/01_pronosticos/` y entran tal
cual a la evaluación.

`resultados/` también incluye, como referencia, salidas del análisis estructural con el modelo de muestra completa
(coeficientes, respuestas generalizadas, traspaso a precios, GFEVD y sensibilidad), cuyos scripts no forman parte de
este ejemplo.

## Resultados

MAE relativo al paseo aleatorio, objetivos de 2017-01 a 2026-08 (115 por horizonte):

| Modelo | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 |
|---|---:|---:|---:|---:|---:|---:|
| **GTVP-VAR** | **0,875\*** | **0,777\*** | **0,724\*** | **0,655\*** | **0,650\*** | **0,686\*** |
| TVP-VAR | 0,833\*\* | 0,776\*\* | 0,747\* | 0,728\*\* | 0,721\* | 0,753\* |
| BVAR | 0,843\*\* | 0,797\* | 0,769\* | 0,756\* | 0,782\* | 0,878 |
| VAR | 0,874\* | 0,804 | 0,776 | 0,768\* | 0,799 | 0,890 |
| FAVAR | 0,909 | 0,812 | 0,761\* | 0,769\* | 0,799 | 0,884 |
| RF | 0,993 | 0,815 | 0,822 | 0,795 | 0,849 | 0,902 |
| MRF | 0,886 | 0,837\* | 0,791 | 0,874 | 0,836 | 0,885 |

Giacomini-White contra el paseo aleatorio, pérdida absoluta: \* p<0,10, \*\* p<0,05, \*\*\* p<0,01. Tablas completas
en [`resultados/02_evaluacion/`](resultados/02_evaluacion).

## Reproducibilidad

Con la misma semilla y las mismas versiones, el código reproduce `resultados/` (las tablas de evaluación, byte a
byte). Entre máquinas distintas puede haber diferencias chicas en los pronósticos: OpenBLAS/LAPACK eligen núcleos
según el procesador y pueden cambiar el último dígito de una cuenta, lo que alcanza para que un árbol elija otro corte
cuando dos candidatos están casi empatados. En la mayoría de los orígenes los pronósticos coinciden a 1e-17; en
algunos difieren hasta en 1e-4 (0,01 pp de inflación mensual).
