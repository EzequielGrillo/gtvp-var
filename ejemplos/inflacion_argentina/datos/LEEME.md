# Datos

El dataset del ejemplo (`Dataset_base.xlsx`) **no se distribuye**. Para correr el ejemplo hay que ubicarlo en esta
carpeta (`ejemplos/inflacion_argentina/datos/Dataset_base.xlsx`). Git lo ignora, para que no se suba por error.

Con el formato de abajo se puede armar la base de nuevo o adaptarla a otro país.

## Formato

- Una hoja con **394 filas mensuales**, de 1993-11 a 2026-08, sin datos faltantes.
- Columna `Fecha`: primer día de cada mes.
- Columna `Y`: igual a `IPC_t` (`config.base()` lo verifica).
- Las **70 columnas del estado S** (abajo), en ese orden; la última es la tendencia `t`.
- Columnas `Desempleo_t-2..t-4` y `CompraRIN_t..t-2`: están en la base pero se descartan al leerla.

La fila de la fecha *t* contiene la información disponible en *t*. Por eso cada serie entra con su rezago de
publicación: las que se conocen en el mes aparecen como `_t, _t-1, _t-2`; las que se publican con dos meses de demora,
como `_t-2, _t-3, _t-4`.

## Variables del VAR (y_t, K = 5)

| Columna | Variable |
|---|---|
| `IPC_t` | inflación mensual: log-variación del IPC |
| `TC_t` | log-variación del tipo de cambio oficial |
| `Brecha_t` | brecha cambiaria: log(CCL / oficial) |
| `Tasa_t` | tasa de interés |
| `BrechaPIB_t-3` | brecha del producto (entra con 3 meses de rezago) |

La parte lineal es `x_t = [1, y_{t-1}]` (VAR(1), k = 6).

## Estado S (70 columnas)

23 series con 3 versiones cada una (la más reciente disponible y dos rezagos), más la tendencia. La columna "al
iterar" indica cómo se proyecta cada serie en los pronósticos a más de un mes (`replicacion/estado.py`).

| Serie | Columnas | Descripción | Al iterar |
|---|---|---|---|
| IPC | `IPC_t`, `_t-1`, `_t-2` | inflación mensual | pronóstico del VAR |
| TC | `TC_t`, `_t-1`, `_t-2` | variación del tipo de cambio oficial | pronóstico del VAR |
| CCL | `CCL_t`, `_t-1`, `_t-2` | variación del dólar contado con liquidación | identidad: `Brecha_t − Brecha_t-1 + TC_t` |
| Brecha | `Brecha_t`, `_t-1`, `_t-2` | brecha cambiaria | pronóstico del VAR |
| Tasa | `Tasa_t`, `_t-1`, `_t-2` | tasa de interés | pronóstico del VAR |
| TCR | `TCR_t`, `_t-1`, `_t-2` | tipo de cambio real | identidad: `TCR_t-1 + TC_t − IPC_t` |
| RRII | `RRII_t`, `_t-1`, `_t-2` | reservas internacionales | identidad: `RRII_t-1 + varRIN_t` |
| T10Y | `T10Y_t`, `_t-1`, `_t-2` | tasa del Tesoro de EE. UU. a 10 años | exógena: promedio de 12 meses |
| M2 | `M2_t-2`, `_t-3`, `_t-4` | agregado monetario M2 | BVARX satélite |
| BM | `BM_t-2`, `_t-3`, `_t-4` | base monetaria | BVARX satélite |
| X | `X_t-2`, `_t-3`, `_t-4` | exportaciones | BVARX satélite |
| Q | `Q_t-2`, `_t-3`, `_t-4` | importaciones | BVARX satélite |
| BC | `BC_t-2`, `_t-3`, `_t-4` | balanza comercial | identidad: `exp(X) − exp(Q)` |
| Sal | `Sal_t-2`, `_t-3`, `_t-4` | salarios | BVARX satélite |
| BrechaPIB | `BrechaPIB_t-3`, `_t-4`, `_t-5` | brecha del producto | pronóstico del VAR |
| Deficit | `Deficit_t-2`, `_t-3`, `_t-4` | resultado fiscal | exógena: promedio de 12 meses |
| MERVAL | `MERVAL_t`, `_t-1`, `_t-2` | índice bursátil MERVAL | BVARX satélite |
| M2PIB | `M2PIB_t-2`, `_t-3`, `_t-4` | M2 sobre PIB | BVARX satélite |
| IPMP | `IPMP_t`, `_t-1`, `_t-2` | índice de precios de materias primas | exógena: promedio de 12 meses |
| vIPMP | `vIPMP_t`, `_t-1`, `_t-2` | variación del índice de precios de materias primas | exógena: promedio de 12 meses |
| IPI | `IPI_t-1`, `_t-2`, `_t-3` | índice de producción industrial | BVARX satélite |
| CredReal | `CredReal_t`, `_t-1`, `_t-2` | crédito real | BVARX satélite |
| varRIN | `varRIN_t`, `_t-1`, `_t-2` | variación de las reservas | BVARX satélite |
| t | `t` | tendencia (suma 1 por mes) | `t + 1` |

Orden exacto de las columnas:

```
IPC_t IPC_t-1 IPC_t-2 TC_t TC_t-1 TC_t-2 CCL_t CCL_t-1 CCL_t-2 Brecha_t Brecha_t-1 Brecha_t-2 Tasa_t Tasa_t-1 Tasa_t-2
TCR_t TCR_t-1 TCR_t-2 RRII_t RRII_t-1 RRII_t-2 T10Y_t T10Y_t-1 T10Y_t-2 M2_t-2 M2_t-3 M2_t-4 BM_t-2 BM_t-3 BM_t-4
X_t-2 X_t-3 X_t-4 Q_t-2 Q_t-3 Q_t-4 BC_t-2 BC_t-3 BC_t-4 Sal_t-2 Sal_t-3 Sal_t-4 BrechaPIB_t-3 BrechaPIB_t-4
BrechaPIB_t-5 Deficit_t-2 Deficit_t-3 Deficit_t-4 MERVAL_t MERVAL_t-1 MERVAL_t-2 M2PIB_t-2 M2PIB_t-3 M2PIB_t-4
IPMP_t IPMP_t-1 IPMP_t-2 vIPMP_t vIPMP_t-1 vIPMP_t-2 IPI_t-1 IPI_t-2 IPI_t-3 CredReal_t CredReal_t-1 CredReal_t-2
varRIN_t varRIN_t-1 varRIN_t-2 t
```

En la base, las columnas de `Desempleo` van después de las de `MERVAL`, y las de `CompraRIN` después de las de `M2PIB`.
Como se descartan, el orden del estado queda como arriba.
