# Comparador de modelos predictivos: Random Forest vs. XGBoost para la flotación de relaves de lixiviación

Aplicación en **R + Shiny** (con versión equivalente en **Python para Google Colab**) que entrena, valida y compara modelos de *machine learning* para predecir tres variables del circuito de flotación de relaves de lixiviación de una planta concentradora de oro, y contrasta las hipótesis de investigación con criterios estadísticos fijos.

> Proyecto desarrollado como parte de la tesis de maestría **«Aplicación de machine learning para predecir la flotación de relaves de lixiviación, Lima 2026»** — Maestría en Inteligencia Artificial, Universidad César Vallejo.
> Autores: **Barboza Duran, Edwin Miguel** · **Ruiz Torres, Alfonso**. Asesor: MSc. Ruben Quispe Llacctarimay.

<!-- Sustituye <usuario>/<repositorio> por los de tu cuenta para activar el botón de Colab -->
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<usuario>/<repositorio>/blob/main/python/Comparador_RF_XGBoost_Flotacion_v7.ipynb)

---

## Contenido

- [Objetivo](#objetivo)
- [Variables objetivo y predictores](#variables-objetivo-y-predictores)
- [Protocolo de modelado](#protocolo-de-modelado)
- [Hipótesis y criterios de decisión](#hipótesis-y-criterios-de-decisión)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación y ejecución (R + Shiny)](#instalación-y-ejecución-r--shiny)
- [Ejecución en Google Colab (Python)](#ejecución-en-google-colab-python)
- [Formato de los datos](#formato-de-los-datos)
- [Vistas de la aplicación](#vistas-de-la-aplicación)
- [Resultados con los datos de la tesis](#resultados-con-los-datos-de-la-tesis)
- [Documentación](#documentación)
- [Limitaciones](#limitaciones)
- [Cómo citar](#cómo-citar)
- [Licencia](#licencia)

---

## Objetivo

Construir sensores blandos (*soft sensors*) que estimen, a partir de los registros operacionales y de laboratorio de cada guardia, tres resultados de la flotación de relaves de lixiviación, y determinar:

1. si los modelos de IA predicen esas variables mejor que una referencia ingenua (la media histórica), y
2. cuál de los dos algoritmos, **Random Forest** o **XGBoost**, presenta un desempeño significativamente superior.

## Variables objetivo y predictores

| Dimensión | Variable objetivo | Unidad |
| --- | --- | --- |
| D1. Rendimiento de recuperación metálica | Recuperación de Au | % |
| D2. Calidad del concentrado | Ley de Au en el concentrado | g/t |
| D3. Eficiencia en el uso de insumos químicos | Consumo específico de reactivos | kg/t |

**Predictores (38 variables en 5 bloques):** composición de la alimentación (leyes de Cu, Pb, Zn, Fe, As y Ag, ley de Au de cabeza, granulometría, K30/K50/K80…), dosificación de reactivos (sulfato de cobre, PAX + SIPX, MIBC, metabisulfito de sodio, antiespumante), variables fisicoquímicas (pH, ORP, oxígeno disuelto, cianuro libre), operacionales e hidráulicas (masa, caudal de pulpa, tiempo de flotación…) y de aireación y espuma.

Las cinco dosificaciones de reactivos se usan siempre como predictores de las tres variables; el consumo específico total se usa como predictor de la recuperación y de la ley (nunca de sí mismo).

## Protocolo de modelado

```
Excel/CSV ─► limpieza y etiquetas ─► partición temporal 70/15/15 ─► por cada variable objetivo:
   ├─ Validación: RF y XGBoost ajustados solo con entrenamiento
   ├─ Prueba: reentreno con entrenamiento + validación, predicción ÚNICA del conjunto de prueba
   └─ Evaluación: vs. modelo base, criterios, ranking y prueba Diebold-Mariano (e² y |e|)
─► contraste de hipótesis (Tablas 8 y 9) ─► app de 5 pestañas + informe PDF
```

- **Partición temporal 70 / 15 / 15** (entrenamiento / validación / prueba): con 547 registros, 382 / 82 / 83. Los registros más recientes quedan para prueba, lo que evita fuga de información. Si el Excel trae una columna `Particion`, se respeta.
- **Modelo base:** media histórica del conjunto de ajuste (entrenamiento + validación), equivalente a `DummyRegressor(strategy="mean")`.
- **Imputación sin fuga:** los faltantes se rellenan con la media calculada solo en el conjunto de ajuste.
- **Hiperparámetros:** un juego por variable objetivo (`HP_TESIS`), elegido por el error en validación.
- **Métricas:** R², MAE, RMSE, MAPE y RRMSE; reducción de MAE y RMSE frente al modelo base como indicador descriptivo.
- **Prueba estadística:** Diebold-Mariano con corrección de Harvey, Leybourne y Newbold (HLN), aplicada con errores al cuadrado y con errores absolutos.
- **Interpretabilidad:** importancia de variables %IncMSE (Random Forest) y Gain (XGBoost).

## Hipótesis y criterios de decisión

Todas se evalúan en el **conjunto de prueba**.

| Hipótesis | Se acepta Hi cuando… |
| --- | --- |
| **Ho1-Hi1** · Recuperación de Au | al menos un algoritmo obtiene a la vez R² ≥ 0.75, MAPE ≤ 10 % y MAE y RMSE menores que los del modelo base |
| **Ho2-Hi2** · Ley de Au en el concentrado | ídem |
| **Ho3-Hi3** · Consumo específico de reactivos | ídem |
| **Ho-Hi** · General | se aceptan Hi1, Hi2 y Hi3 |
| **Ho4-Hi4** · RF vs. XGBoost | en al menos una variable, un algoritmo tiene mayor R² y menores RMSE, MAE y MAPE que el otro **y** la prueba Diebold-Mariano da p < 0.05 tanto con errores al cuadrado como con errores absolutos |

En cualquier otro caso se acepta la hipótesis nula correspondiente. La app y el informe PDF muestran el contraste en dos tablas:

- **Tabla 8** — Ho1-Hi1 a Ho3-Hi3: métricas del modelo base, Random Forest y XGBoost por variable, «Cumple criterios» por modelo y una decisión por variable; debajo, la decisión de Ho-Hi.
- **Tabla 9** — Ho4-Hi4: métricas de RF y XGBoost, p-valores DM (e² y |e|) y resultado por variable; debajo, la decisión global.

## Estructura del repositorio

Estructura sugerida (ajústala a tus carpetas):

```
.
├── R/
│   └── S3_TESIS_MIA_BarbozaEdwin-RuizAlfonso_Codigo_RStudio_v7.R   # app Shiny (versión oficial)
├── python/
│   └── Comparador_RF_XGBoost_Flotacion_v7.ipynb                    # versión para Google Colab
├── docs/
│   ├── Guia_codigo_Comparador_RF_XGBoost_v7.docx                   # cómo funciona el código, con vistas
│   ├── Diccionario_codigo_Comparador_RF_XGBoost_v7.docx            # librerías, funciones, variables
│   ├── Matriz_metodologica_tesis_flotacion_v7.docx                 # diseño, matriz de consistencia, hipótesis
│   └── informe_comparacion_ejemplo.pdf                             # informe PDF generado por la app
├── img/                                                            # capturas para este README
├── data/
│   └── README.md                                                   # descripción del formato (sin datos)
└── README.md
```

> ⚠️ **Datos confidenciales.** El archivo `Datos_Flotacion_de_Relaves.xlsx` contiene información operacional de la empresa titular y **no debe publicarse**. Agrega `data/*.xlsx` y `data/*.csv` a tu `.gitignore`.

## Instalación y ejecución (R + Shiny)

**Requisitos:** R ≥ 4.2 (la tesis usó R 4.6.1) y RStudio.

```r
install.packages(c("shiny", "bslib", "DT", "ggplot2", "randomForest",
                   "xgboost", "gridExtra", "readxl"))
```

`grid`, `stats`, `utils`, `grDevices` y `tools` vienen con R; `gtable` se instala junto con `ggplot2`. Si falta algún paquete, la app se detiene y muestra el comando `install.packages()` exacto.

**Ejecución:**

1. Abre `R/S3_TESIS_MIA_BarbozaEdwin-RuizAlfonso_Codigo_RStudio_v7.R` en RStudio y pulsa **Run App** (o `shiny::runApp("ruta/al/script.R")`).
2. En la barra lateral, **Cargar dataset** → elige el Excel o CSV.
3. Revisa las variables objetivo y predictoras (se preseleccionan las de la tesis) y, si quieres, las **Opciones avanzadas** (partición, faltantes, hiperparámetros).
4. Pulsa **Ejecutar protocolo**.
5. Revisa las pestañas y descarga el **informe PDF** (Portada, Datos y Resultados).

> La lectura del Excel requiere una sesión de R en UTF-8. En RStudio sobre Windows con R ≥ 4.2 funciona sin ajustes; en Linux o macOS, si ves errores de codificación, inicia R con `LANG=C.UTF-8`.

**Navegar el código:** cada bloque termina en `----`, así que aparece en el panel *Outline* de RStudio (Ctrl+Shift+O). El script está organizado en 16 secciones: paquetes, constantes, lectura de datos, métricas y prueba DM, descriptivos, imputación, ajuste de modelos, entrenamiento por objetivo, partición, contraste de hipótesis, gráficos, tema visual, componentes de interfaz, UI, servidor y lanzamiento.

## Ejecución en Google Colab (Python)

1. Abre `python/Comparador_RF_XGBoost_Flotacion_v7.ipynb` en Colab (botón de arriba o *Archivo → Subir cuaderno*).
2. **Entorno de ejecución → Ejecutar todo.**
3. **Paso 1:** elige *Subir desde mi computadora* o *Google Drive* (escribe la ruta del Excel).
4. **Paso 2:** revisa los parámetros del formulario y ejecuta el protocolo; se muestran las Tablas 8 y 9.
5. **Paso 3:** panel con las pestañas Portada, Datos, Resultados, Discusión y Conclusiones.
6. **Paso 4:** genera y descarga el informe PDF (si cargaste desde Drive, también se guarda junto al Excel).

Librerías: `pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `matplotlib` y `openpyxl` (el cuaderno instala las que falten).

> La versión de Python reproduce el protocolo, las hipótesis y las vistas de la app R, pero las cifras difieren ligeramente por las diferencias entre librerías (`randomForest` vs. `scikit-learn`, generadores aleatorios). **La versión en R es la referencia oficial de la tesis.**

## Formato de los datos

Un Excel (`.xlsx`, primera hoja) o CSV con **un registro por guardia**. Los encabezados de planta se traducen automáticamente a nombres internos mediante la tabla `alias` de la sección 3.3 del script, por ejemplo:

| Encabezado en el Excel | Nombre interno |
| --- | --- |
| `R.Au ccPy (%)` | `recuperacion_au` |
| `L.Au ccPy (g/t)` | `ley_concentrado_au` |
| `Consumo específico total (kg/t)` | `consumo_reactivos` |
| `CuSO4·5H2O (kg/t)`, `Na2S2O5 (kg/t)`, `Z6+Z11 (kg/t)`, `MIBC (kg/t)`, `Antiespumante (kg/t)` | dosificaciones de reactivos |
| `Fecha`, `Guardia` | orden cronológico de la partición |
| `Particion` *(opcional)* | Entrenamiento / Validación / Prueba predefinidos |

Las demás columnas numéricas se ofrecen como predictores; se excluyen automáticamente las de metadatos (año, mes, id, índice…). Si tu Excel usa otro encabezado, añade su alias en `normalizar_columnas_excel()`.

## Vistas de la aplicación

<!-- Copia en img/ las capturas que quieras mostrar; por ejemplo: -->
| Portada | Contraste de hipótesis (Tabla 8) |
| --- | --- |
| ![Portada](img/portada.png) | ![Tabla 8](img/tabla8.png) |

| Pestaña | Contenido |
| --- | --- |
| **Portada** | Indicadores del mejor modelo por variable, protocolo, hipótesis general y veredicto APTO / NO APTO |
| **Datos** | Tamaño de la partición, estadísticos descriptivos y boxplot por subconjunto, dataset cargado |
| **Resultados** | Tabla 4 (métricas de prueba), Tablas 5–7 (por subconjunto), Tablas 8 y 9 (hipótesis), Figuras 1–7 (R², real vs. predicho, importancia de variables) |
| **Discusión** / **Conclusiones** | Texto fijo del informe de tesis |

## Resultados con los datos de la tesis

Conjunto de prueba (n = 83), versión R v7:

| Variable | Random Forest R² / MAPE | XGBoost R² / MAPE | Hipótesis | Comparación RF vs. XGBoost |
| --- | --- | --- | --- | --- |
| Recuperación de Au (%) | 0.973 / 0.94 % | 0.960 / 1.15 % | Se acepta Hi1 | RF significativamente superior (p = 0.007 e²; p = 0.014 \|e\|) |
| Ley de Au en el concentrado (g/t) | 0.985 / 0.46 % | 0.979 / 0.53 % | Se acepta Hi2 | Sin superioridad significativa (p = 0.080 con \|e\|) |
| Consumo específico de reactivos (kg/t) | 0.740 / 14.70 % | 0.897 / 8.92 % | Se acepta Hi3 (solo XGBoost cumple) | XGBoost significativamente superior (p < 0.001) |

**Decisiones:** se acepta **Hi** (hipótesis general) y se acepta **Hi4**.

## Documentación

| Documento | Contenido |
| --- | --- |
| Guía del código | Finalidad, flujo, recorrido por las vistas con capturas y explicación por bloques |
| Diccionario del código | Librerías, 68 funciones propias, constantes, controles, salidas y estructuras de resultados, con su línea en el script |
| Matriz metodológica | Diseño de investigación, procedimiento por fases, matriz de consistencia, planteamiento Ho/Hi y operacionalización de variables |

## Limitaciones

- Datos de una sola planta y de un periodo de nueve meses (enero–setiembre de 2026).
- El conjunto de prueba (n = 83) abarca un único periodo de operación.
- El consumo específico total es la suma de las cinco dosificaciones; el modelo de consumo mide la capacidad de reconstruirlo a partir de sus componentes.
- Las medidas de importancia no indican el sentido del efecto, y no se estiman intervalos de predicción.
- Los resultados de XGBoost pueden variar ligeramente entre versiones del paquete; registra la tuya con `packageVersion("xgboost")`.

## Cómo citar

```
Barboza Duran, E. M., & Ruiz Torres, A. (2026). Aplicación de machine learning para predecir
la flotación de relaves de lixiviación, Lima 2026 [Tesis de maestría, Universidad César Vallejo].
Código disponible en https://github.com/<usuario>/<repositorio>
```

## Licencia

<!-- Elige una licencia y agrega el archivo LICENSE. MIT es habitual para código académico. -->
Código distribuido bajo la licencia **MIT** (ver `LICENSE`). Los datos operacionales de la planta no forman parte del repositorio y son propiedad de la empresa titular.
