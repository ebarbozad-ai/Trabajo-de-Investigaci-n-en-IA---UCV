# ============================================================================
# Comparador de modelos predictivos: Random Forest y XGBoost  (versión v7)
# Tesis: Aplicación de machine learning para predecir la flotación de relaves
#        de lixiviación, Lima 2026 — Barboza Duran, E. M. & Ruiz Torres, A. (UCV)
#
# ============================================================================
# 1. PAQUETES ----
# Comprueba que estén instalados los 8 paquetes; si falta alguno, detiene la
# app y muestra el comando install.packages() exacto. Luego los carga.
# ============================================================================

paquetes_necesarios <- c("shiny", "bslib", "DT", "ggplot2", "randomForest",
                         "xgboost", "gridExtra", "readxl")
faltantes <- paquetes_necesarios[!sapply(paquetes_necesarios, requireNamespace, quietly = TRUE)]
if (length(faltantes) > 0) {
  stop("Faltan paquetes:\ninstall.packages(c(",
       paste(sprintf('"%s"', faltantes), collapse = ", "), "))")
}

# Carga de librerías
library(shiny); library(bslib); library(DT); library(ggplot2)
library(randomForest); library(xgboost)
library(grid); library(gridExtra); library(readxl)

# ============================================================================
# 2. DICCIONARIO Y CONSTANTES ----
# Valores fijos del protocolo. Si cambias aquí un umbral o un hiperparámetro,
# cambia en toda la app (tarjetas, tablas, criterios e informe PDF).
# ============================================================================
## 2.1 Diccionario de etiquetas ----
# Nombre interno de cada columna -> texto que se muestra en pantalla (con unidad).
ETIQUETAS_TESIS <- c(
  masa_alimentacion   = "Masa de alimentación (t)",
  ley_alimentacion_au = "Ley de Au de cabeza (g/t)",
  densidad_pulpa      = "Densidad de pulpa (g/L)",
  pct_solidos         = "Porcentaje de sólidos (%)",
  granulometria       = "Granulometría -200 mallas (%)",
  ph_inicial          = "pH inicial",
  ph_final            = "pH final",
  ph                  = "pH",
  potencial_redox     = "Potencial redox (mV)",
  caudal_alimentacion = "Caudal de pulpa (m³/h)",
  tiempo_residencia   = "Tiempo de flotación (min)",
  sulfato_cobre       = "Sulfato de cobre (kg/t)",
  metabisulfito_sodio = "Metabisulfito de sodio (kg/t)",
  pax_sipx            = "PAX + SIPX (Z-6 + Z-11) (kg/t)",
  mibc                = "MIBC (kg/t)",
  antiespumante       = "Antiespumante (kg/t)",
  recuperacion_au     = "Recuperación de Au (%)",
  ley_concentrado_au  = "Ley de Au en el concentrado (g/t)",
  consumo_reactivos   = "Consumo específico de reactivos (kg/t)",
  l_cu_ccpy = "L.Cu ccPy (g/t)", l_pb_ccpy = "L.Pb ccPy (g/t)",
  l_zn_ccpy = "L.Zn ccPy (g/t)", l_fe_ccpy = "L.Fe ccPy (g/t)",
  l_as_ccpy = "L.As ccPy (g/t)", l_ag_ccpy = "L.Ag ccPy (g/t)",
  h2o_ccpy = "%H2O ccPy (%)", m200_ccpy = "200M_ccPy (%)",
  k30_um = "K30 (µm)", k50_um = "K50 (µm)", k80_um = "K80 (µm)",
  o2_disuelto = "O2 disuelto (mg/L)",
  fcn_inic = "FCN_Inic (ppm)", fcn_fin = "FCN_Fin (ppm)",
  cn_libre_residual = "CN libre residual (mg/L)",
  q_agua_lavado = "Q agua de lavado (m³/h)",
  nivel_pulpa = "Nivel de pulpa (%)",
  caudal_aire = "Caudal de aire (Nm³/h)",
  presion_aire = "Presión de aire (kPa)",
  presion_manifold = "Presión manifold (kPa)",
  apertura_aire = "Apertura de aire (%)",
  altura_espuma = "Altura de espuma (mm)",
  Fecha = "Fecha", Guardia = "Guardia", Particion = "Particion"
)
## 2.2 Objetivos, umbrales y reactivos ----
# OBJETIVOS_TESIS : variables objetivo que se preseleccionan (Ho1-Hi1, Ho2-Hi2, Ho3-Hi3).
# UMBRAL_*        : criterios de aceptación de la tesis.
# REACTIVOS_TESIS : dosificaciones que siempre entran como predictores (v3).
# PATRON_META     : columnas numéricas que NO son predictores (año, mes, id...).
OBJETIVOS_TESIS        <- c("recuperacion_au", "ley_concentrado_au", "consumo_reactivos")
UMBRAL_R2_TESIS        <- 0.75
UMBRAL_MAPE_TESIS      <- 10
UMBRAL_P_TESIS         <- 0.05
REACTIVOS_TESIS        <- c("sulfato_cobre", "metabisulfito_sodio", "pax_sipx", "mibc", "antiespumante")
PATRON_META            <- "^(a.o|anio|year|mes|month|semana|week|dia|day|id|codigo|indice|index|orden|fila)$"

## 2.3 Hiperparámetros de la tesis ----
# Un juego por objetivo. RF: ntree (árboles), max_depth, nodesize (tamaño mínimo
# de nodo), max_features (fracción de variables por división).
# XGB: nrounds (árboles), eta (tasa de aprendizaje), max_depth, subsample y
# colsample (fracción de filas y columnas por árbol).
HP_TESIS <- list(
  recuperacion_au = list(
    rf  = list(ntree = 300, max_depth = 8,  nodesize = 5,  max_features = 0.6),
    xgb = list(nrounds = 500, eta = 0.01, max_depth = 3, subsample = 0.9, colsample = 0.9)),
  ley_concentrado_au = list(
    rf  = list(ntree = 300, max_depth = NA, nodesize = 1,  max_features = 1.0),
    xgb = list(nrounds = 200, eta = 0.10, max_depth = 5, subsample = 0.7, colsample = 0.9)),
  consumo_reactivos = list(
    rf  = list(ntree = 300, max_depth = 8,  nodesize = 10, max_features = 0.6),
    xgb = list(nrounds = 200, eta = 0.03, max_depth = 3, subsample = 0.7, colsample = 0.6))
)

## 2.4 Colores y rangos cualitativos ----
# Azul = Random Forest, naranja = XGBoost, gris = modelo base. Las funciones
# RANGOS_* convierten R², MAPE y RRMSE en una categoría verbal.
COLORES_MODELO <- c(
  "Random Forest" = "#2A78D6",
  "XGBoost"       = "#EB6834",
  "Modelo base"   = "#D1D5DB"
)

RANGOS_R2 <- function(x) {
  if (!is.finite(x)) return("—")
  if (x >= 0.75) "Sustancial" else if (x >= 0.50) "Moderado"
  else if (x >= 0.25) "Débil" else "Insuficiente"
}
RANGOS_MAPE <- function(x) {
  if (!is.finite(x)) return("—")
  if (x < 10) "Excelente" else if (x < 20) "Bueno"
  else if (x < 50) "Aceptable" else "Inexacto"
}
RANGOS_RRMSE <- function(x) {
  if (!is.finite(x)) return("—")
  if (x < 10) "Excelente" else if (x < 20) "Bueno"
  else if (x < 30) "Regular" else "Deficiente"
}

# ============================================================================
# 3. LECTURA Y LIMPIEZA DE DATOS ----
# Convierte cualquier Excel/CSV de planta en una tabla con nombres internos
# estándar y etiquetas legibles. Se usa al pulsar «Cargar dataset».
# ============================================================================
## 3.1 Lectura de CSV ----
# Detecta si el separador es coma o punto y coma (CSV en español).
leer_csv_robusto <- function(ruta) {
  lineas <- readLines(ruta, encoding = "UTF-8", warn = FALSE)
  if (length(lineas) == 0) stop("El archivo está vacío.")
  lineas[1] <- sub("^\ufeff", "", lineas[1])
  texto <- paste(lineas, collapse = "\n")
  primera <- lineas[1]
  n_coma <- lengths(regmatches(primera, gregexpr(",", primera)))
  n_puntocoma <- lengths(regmatches(primera, gregexpr(";", primera)))
  if (n_puntocoma > n_coma) {
    read.csv2(text = texto, check.names = FALSE, stringsAsFactors = FALSE, encoding = "UTF-8")
  } else {
    read.csv(text = texto, check.names = FALSE, stringsAsFactors = FALSE, encoding = "UTF-8")
  }
}

## 3.2 Limpieza de nombres y etiquetas ----
# limpiar_texto(): quita saltos de línea y espacios dobles de los encabezados.
# sanear_nombres(): nombres válidos para R + guarda la etiqueta original.
# etiqueta() / etiquetas_de(): devuelven el texto legible de una columna.
limpiar_texto <- function(x) {
  x <- gsub("[\r\n\t]+", " ", as.character(x))
  trimws(gsub(" {2,}", " ", x))
}

sanear_nombres <- function(df) {
  names(df) <- limpiar_texto(names(df))
  originales <- names(df)
  seguros <- make.names(originales, unique = TRUE)
  names(df) <- seguros
  et <- ifelse(originales %in% names(ETIQUETAS_TESIS), ETIQUETAS_TESIS[originales], originales)
  attr(df, "etiquetas") <- stats::setNames(unname(et), seguros)
  df
}

etiqueta <- function(datos, nombre) {
  et <- attr(datos, "etiquetas")
  if (!is.null(et) && !is.null(nombre) && nombre %in% names(et)) unname(et[[nombre]]) else nombre
}
etiquetas_de <- function(datos, nombres) {
  vapply(nombres, function(n) etiqueta(datos, n), character(1))
}

## 3.3 Traducción de encabezados de planta ----
# normalizar_clave(): 'L.Au ccPy (g/t)' -> 'lauccpygt' (sin tildes ni símbolos).
# normalizar_columnas_excel(): busca esa clave en la tabla 'alias' y renombra.
# >> Si agregas una columna nueva al Excel con otro encabezado, añade aquí su alias.
normalizar_clave <- function(x) {
  x <- tolower(limpiar_texto(x))
  x <- chartr("áéíóúüñàèìòù", "aeiouunaeiou", x)
  gsub("[^a-z0-9]+", "", x)
}

normalizar_columnas_excel <- function(df) {
  alias <- c("fecha"="Fecha", "mes"="Mes", "anio"="anio", "ano"="anio", "guardia"="Guardia",
             "lauccpygt"="ley_concentrado_au", "na2s2o5kgt"="metabisulfito_sodio",
             "masadesolidostms"="masa_alimentacion", "laucbzgtt"="ley_alimentacion_au",
             "laucbzgt"="ley_alimentacion_au", "densidaddepulpagl"="densidad_pulpa",
             "solidos"="pct_solidos", "granulometria200mesh"="granulometria",
             "phinic"="ph_inicial", "phfin"="ph_final", "ph"="ph", "orpmv"="potencial_redox",
             "qpm3h"="caudal_alimentacion", "qp"="caudal_alimentacion",
             "tiempoflotmin"="tiempo_residencia", "cuso45h2okgt"="sulfato_cobre",
             "nas2o5kgt"="metabisulfito_sodio", "z6z11kgt"="pax_sipx", "mibckgt"="mibc",
             "antiespumantekgt"="antiespumante", "rauccpy"="recuperacion_au",
             "lauccpy"="ley_concentrado_au", "consumoespecificototalkgt"="consumo_reactivos",
             "particionmodelado"="Particion", "particion"="Particion")
  originales <- names(df)
  claves <- normalizar_clave(originales)
  names(df) <- ifelse(claves %in% names(alias), unname(alias[claves]), originales)
  df
}

## 3.4 Lectura de Excel y punto de entrada ----
# leer_excel_robusto(): lee la primera hoja y corta columnas vacías a la derecha.
# leer_archivo_robusto(): decide por extensión (Excel o CSV); lo usa el servidor.
leer_excel_robusto <- function(ruta) {
  hojas <- readxl::excel_sheets(ruta)
  if (length(hojas) == 0) stop("El libro de Excel no contiene hojas.")
  hoja <- hojas[[1]]
  df <- readxl::read_excel(ruta, sheet = hoja, .name_repair = "minimal", guess_max = 10000)
  if (nrow(df) == 0 || ncol(df) == 0) stop("La hoja de Excel está vacía.")
  vacias <- which(is.na(names(df)) | !nzchar(trimws(names(df))))
  if (length(vacias) > 0 && vacias[[1]] > 1) df <- df[, seq_len(vacias[[1]] - 1), drop = FALSE]
  df <- normalizar_columnas_excel(as.data.frame(df, stringsAsFactors = FALSE))
  attr(df, "hoja_excel") <- hoja
  df
}

leer_archivo_robusto <- function(ruta, nombre = "") {
  ext <- tolower(tools::file_ext(if (nzchar(nombre)) nombre else ruta))
  if (ext %in% c("xlsx", "xls")) leer_excel_robusto(ruta) else leer_csv_robusto(ruta)
}

# ============================================================================
# 4. MÉTRICAS, PRUEBA DIEBOLD-MARIANO Y CRITERIOS ----
# Cómo se mide cada modelo y cómo se decide si cumple los criterios de la tesis.
# ============================================================================
## 4.1 Métricas de error ----
# R², MAE, RMSE, MAPE (ignora valores reales ~ 0) y RRMSE (RMSE / media).
calcular_metricas <- function(real, pred) {
  real <- as.numeric(real); pred <- as.numeric(pred)
  ok <- is.finite(real) & is.finite(pred)
  real <- real[ok]; pred <- pred[ok]
  if (length(real) == 0) {
    return(data.frame(R2 = NA_real_, MAE = NA_real_, RMSE = NA_real_, MAPE = NA_real_,
                      RRMSE = NA_real_, n = 0L))
  }
  residuos <- real - pred
  rmse <- sqrt(mean(residuos^2)); mae <- mean(abs(residuos))
  ss_res <- sum(residuos^2); ss_tot <- sum((real - mean(real))^2)
  r2 <- if (ss_tot > .Machine$double.eps) 1 - ss_res / ss_tot else NA_real_
  tol <- max(1e-8, max(abs(real), na.rm = TRUE) * 1e-6)
  idx_validos <- abs(real) > tol
  mape <- if (any(idx_validos)) mean(abs(residuos[idx_validos] / real[idx_validos])) * 100 else NA_real_
  media <- mean(real)
  rrmse <- if (is.finite(media) && abs(media) > tol) 100 * rmse / abs(media) else NA_real_
  data.frame(R2 = r2, MAE = mae, RMSE = rmse, MAPE = mape,
             RRMSE = rrmse, n = length(real))
}

## 4.2 Prueba Diebold-Mariano (HLN) ----
# Compara los errores de prueba de RF (e1) y XGBoost (e2). DM > 0 => RF tiene más
# error. perdida = "cuadratica" usa e², "absoluta" usa |e|. Se calcula con ambas
# para la Tabla 9 y la decisión de Ho4-Hi4.
prueba_dm <- function(e1, e2, h = 1, perdida = c("cuadratica", "absoluta")) {
  perdida <- match.arg(perdida)
  ok <- is.finite(e1) & is.finite(e2)
  L <- if (perdida == "cuadratica") function(e) e^2 else abs
  d <- L(as.numeric(e1[ok])) - L(as.numeric(e2[ok]))
  n <- length(d)
  if (n < 3 || !is.finite(h) || h < 1 || h >= n) {
    return(list(estadistico = NA_real_, p_valor = NA_real_, n = n,
                interpretacion = "Muestra insuficiente para DM-HLN"))
  }
  media_d <- mean(d); var_d <- stats::var(d)
  if (!is.finite(var_d) || var_d <= .Machine$double.eps) {
    return(list(estadistico = NA_real_, p_valor = NA_real_, n = n,
                interpretacion = "No estimable: varianza de la diferencia nula"))
  }
  dm <- media_d / sqrt(var_d / n)
  factor_hln <- sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
  dm_hln <- dm * factor_hln
  p <- 2 * stats::pt(-abs(dm_hln), df = n - 1)
  list(estadistico = dm_hln, p_valor = p, n = n,
       interpretacion = if (is.finite(p) && p < 0.05)
         "Diferencia estadísticamente significativa" else
           "No se detecta diferencia estadísticamente significativa")
}

## 4.3 Criterios de cumplimiento ----
# Devuelve "Sí"/"No" (columna «Cumple criterios» de la Tabla 4 y decisión de
# Ho1-Hi1, Ho2-Hi2 y Ho3-Hi3 en la Tabla 8). Igual para las tres variables:
# R² >= 0.75, MAPE <= 10 % y MAE y RMSE menores que los del modelo base.
evaluar_criterios <- function(m, base) {
  ok <- all(is.finite(c(m$R2, m$MAPE, m$MAE, m$RMSE, base$MAE, base$RMSE))) &&
    m$R2 >= UMBRAL_R2_TESIS && m$MAPE <= UMBRAL_MAPE_TESIS &&
    m$MAE < base$MAE && m$RMSE < base$RMSE
  if (isTRUE(ok)) "Sí" else "No"
}

# ============================================================================
# 5. ESTADÍSTICOS DESCRIPTIVOS ----
# Alimenta la pestaña Datos (tabla y boxplot) y la sección Datos del PDF.
# ============================================================================
descriptivos_por_subconjunto <- function(train, valid, test, variables) {
  calcular <- function(d, nombre_sub) {
    if (is.null(d) || nrow(d) == 0) return(NULL)
    do.call(rbind, lapply(variables, function(v) {
      z <- suppressWarnings(as.numeric(d[[v]]))
      z <- z[is.finite(z)]
      data.frame(
        Subconjunto = nombre_sub, Variable = v,
        n = length(z), n_NA = sum(!is.finite(suppressWarnings(as.numeric(d[[v]])))),
        Media = if (length(z)) mean(z) else NA_real_,
        DE = if (length(z) > 1) stats::sd(z) else NA_real_,
        Min = if (length(z)) min(z) else NA_real_,
        Q1 = if (length(z)) stats::quantile(z, 0.25, names = FALSE) else NA_real_,
        Mediana = if (length(z)) stats::median(z) else NA_real_,
        Q3 = if (length(z)) stats::quantile(z, 0.75, names = FALSE) else NA_real_,
        Max = if (length(z)) max(z) else NA_real_,
        CV_pct = if (length(z) > 1 && mean(z) != 0) 100 * stats::sd(z) / abs(mean(z)) else NA_real_,
        stringsAsFactors = FALSE)
    }))
  }
  rbind(calcular(train, "Entrenamiento"), calcular(valid, "Validación"),
        calcular(test, "Prueba"))
}

# ============================================================================
# 6. IMPUTACIÓN DE VALORES FALTANTES ----
# La media se calcula SOLO con el conjunto de ajuste y se aplica a los demás,
# para que la información de validación/prueba no se filtre al modelo.
# ============================================================================
# Calcula el valor de relleno de cada predictor (media del conjunto de referencia o 0).
calcular_imputaciones <- function(referencia, predictores, metodo = "media") {
  valores <- setNames(vector("list", length(predictores)), predictores)
  for (v in predictores) {
    z <- suppressWarnings(as.numeric(referencia[[v]]))
    z <- z[is.finite(z)]
    if (identical(metodo, "cero")) valores[[v]] <- 0
    else if (length(z) == 0) valores[[v]] <- 0
    else valores[[v]] <- mean(z)
  }
  valores
}
# Convierte a numérico y reemplaza NA/Inf por los valores calculados arriba.
imputar <- function(d, predictores, referencia, metodo) {
  valores <- calcular_imputaciones(referencia, predictores, metodo)
  for (v in predictores) {
    if (!is.numeric(d[[v]])) {
      convertida <- suppressWarnings(as.numeric(as.character(d[[v]])))
      if (all(is.na(convertida) & !is.na(d[[v]]))) {
        stop(sprintf("La variable '%s' no es numérica ni convertible.", v))
      }
      d[[v]] <- convertida
    }
    inval <- !is.finite(d[[v]])
    if (any(inval)) d[[v]][inval] <- valores[[v]]
  }
  attr(d, "valores_imputacion") <- valores
  d
}

# ============================================================================
# 7. AJUSTE DE MODELOS (Random Forest y XGBoost) ----
# Traducen los hiperparámetros de HP_TESIS a las funciones de cada paquete.
# ============================================================================
# Random Forest: mtry = max_features x nº de predictores; max_depth se aproxima con
# maxnodes = 2^max_depth (randomForest no tiene profundidad máxima). Semilla 42.
ajustar_rf <- function(x, y, hp) {
  p <- ncol(x); mtry <- max(1, floor(hp$max_features * p))
  maxnodes <- if (is.null(hp$max_depth) || length(hp$max_depth) == 0 ||
                  is.na(hp$max_depth) || hp$max_depth <= 0) NULL else max(2, 2^min(hp$max_depth, 20))
  set.seed(42)
  randomForest(x = x, y = y, ntree = hp$ntree, mtry = mtry,
               nodesize = hp$nodesize, maxnodes = maxnodes, importance = TRUE)
}
# XGBoost: regresión con error cuadrático, 1 hilo y semilla 42.
ajustar_xgb <- function(x, y, hp) {
  dtrain <- xgb.DMatrix(as.matrix(x), label = y)
  set.seed(42)
  xgb.train(
    params = list(objective = "reg:squarederror", eta = hp$eta, max_depth = hp$max_depth,
                  subsample = hp$subsample, colsample_bytree = hp$colsample, nthread = 1),
    data = dtrain, nrounds = hp$nrounds, verbose = 0)
}
predecir_xgb <- function(modelo, x) predict(modelo, xgb.DMatrix(as.matrix(x)))

# ============================================================================
# 8. ENTRENAMIENTO Y EVALUACIÓN POR OBJETIVO ----
# Función central del protocolo. Se ejecuta una vez por cada variable objetivo
# y devuelve métricas, predicciones, modelos y prueba DM (ver pasos 8.1 a 8.7).
# ============================================================================
entrenar_evaluar_target <- function(objetivo, predictores, train, valid, test,
                                    hp_ui, usar_hp_tesis, metodo_na) {
  # 8.1 Quita filas sin valor del objetivo y verifica tamaños mínimos
  train <- train[is.finite(as.numeric(train[[objetivo]])), , drop = FALSE]
  valid <- valid[is.finite(as.numeric(valid[[objetivo]])), , drop = FALSE]
  test  <- test[is.finite(as.numeric(test[[objetivo]])), , drop = FALSE]
  if (nrow(train) < 5) stop(sprintf("Objetivo '%s': <5 obs en entrenamiento.", objetivo))
  if (nrow(test) < 2) stop(sprintf("Objetivo '%s': <2 obs en prueba.", objetivo))
  
  # 8.2 Elige hiperparámetros: los de la tesis o los genéricos de la interfaz
  hp_tesis <- if (usar_hp_tesis) HP_TESIS[[objetivo]] else NULL
  hp_rf  <- if (!is.null(hp_tesis)) hp_tesis$rf  else hp_ui$rf
  hp_xgb <- if (!is.null(hp_tesis)) hp_tesis$xgb else hp_ui$xgb
  origen_hp <- if (!is.null(hp_tesis)) "tesis" else "interfaz"
  
  # 8.3 FASE DE VALIDACIÓN: imputa con la media de entrenamiento y ajusta rf1 y xg1
  #     SOLO con entrenamiento. Sus métricas van a las Tablas 5, 6 y 7.
  tr_i <- imputar(train, predictores, train, metodo_na)
  va_i <- imputar(valid, predictores, train, metodo_na)
  x_tr <- tr_i[, predictores, drop = FALSE]; y_tr <- tr_i[[objetivo]]
  x_va <- va_i[, predictores, drop = FALSE]; y_va <- va_i[[objetivo]]
  
  rf1 <- ajustar_rf(x_tr, y_tr, hp_rf)
  xg1 <- ajustar_xgb(x_tr, y_tr, hp_xgb)
  
  filas <- list()
  agregar <- function(modelo, subconjunto, real, pred) {
    filas[[length(filas) + 1]] <<- cbind(Modelo = modelo, Subconjunto = subconjunto,
                                         calcular_metricas(real, pred))
  }
  agregar("Random Forest", "Entrenamiento", y_tr, predict(rf1, x_tr))
  agregar("XGBoost",       "Entrenamiento", y_tr, predecir_xgb(xg1, x_tr))
  if (nrow(valid) > 0) {
    agregar("Random Forest", "Validación", y_va, predict(rf1, x_va))
    agregar("XGBoost",       "Validación", y_va, predecir_xgb(xg1, x_va))
  }
  
  # 8.4 FASE DE PRUEBA: une entrenamiento + validación ("ajuste"), reentrena rf2 y
  #     xg2 y predice el conjunto de prueba UNA sola vez. El modelo base predice la
  #     media del ajuste para todas las filas.
  ajuste <- rbind(train, valid)
  aj_i <- imputar(ajuste, predictores, ajuste, metodo_na)
  te_i <- imputar(test, predictores, ajuste, metodo_na)
  x_aj <- aj_i[, predictores, drop = FALSE]; y_aj <- aj_i[[objetivo]]
  x_te <- te_i[, predictores, drop = FALSE]; real_test <- te_i[[objetivo]]
  
  rf2 <- ajustar_rf(x_aj, y_aj, hp_rf)
  xg2 <- ajustar_xgb(x_aj, y_aj, hp_xgb)
  pred_rf  <- as.numeric(predict(rf2, x_te))
  pred_xgb <- predecir_xgb(xg2, x_te)
  pred_base <- rep(mean(y_aj), length(real_test))
  
  agregar("Modelo base",   "Prueba", real_test, pred_base)
  agregar("Random Forest", "Prueba", real_test, pred_rf)
  agregar("XGBoost",       "Prueba", real_test, pred_xgb)
  metricas_todas <- do.call(rbind, filas)
  
  # 8.5 Métricas de prueba, reducción frente al modelo base, criterios y rangos
  #     (alimenta la Tabla 4 y las tarjetas de la Portada).
  metricas <- metricas_todas[metricas_todas$Subconjunto == "Prueba", ]
  metricas$Subconjunto <- NULL
  base <- metricas[metricas$Modelo == "Modelo base", ]
  metricas$Red_MAE_vs_base  <- if (is.finite(base$MAE) && base$MAE > 0)
    100 * (base$MAE - metricas$MAE) / base$MAE else NA_real_
  metricas$Red_RMSE_vs_base <- if (is.finite(base$RMSE) && base$RMSE > 0)
    100 * (base$RMSE - metricas$RMSE) / base$RMSE else NA_real_
  media_obs <- mean(real_test)
  metricas$Cumple_criterios <- vapply(seq_len(nrow(metricas)), function(i)
    evaluar_criterios(metricas[i, ], base), character(1))
  metricas$Cumple_criterios[metricas$Modelo == "Modelo base"] <- "—"
  metricas$Rango_R2   <- vapply(metricas$R2,   RANGOS_R2,   character(1))
  metricas$Rango_MAPE <- vapply(metricas$MAPE, RANGOS_MAPE, character(1))
  metricas$Rango_RRMSE <- vapply(metricas$RRMSE, RANGOS_RRMSE, character(1))
  
  # 8.6 Ranking: suma de puestos en RMSE, MAPE y R² (1 = mejor). El de menor suma
  #     es el mejor modelo (fila verde en la Tabla 4; en empate gana Random Forest).
  candidatos <- metricas$Modelo != "Modelo base"
  ranking <- rep(NA_real_, nrow(metricas))
  idx_cand <- which(candidatos)
  if (length(idx_cand) > 0) {
    r_rmse <- rank(metricas$RMSE[idx_cand], ties.method = "min", na.last = "keep")
    r_mape <- rank(metricas$MAPE[idx_cand], ties.method = "min", na.last = "keep")
    r_r2   <- rank(-metricas$R2[idx_cand], ties.method = "min", na.last = "keep")
    n_cand <- length(idx_cand)
    r_rmse[!is.finite(r_rmse)] <- n_cand + 1
    r_mape[!is.finite(r_mape)] <- n_cand + 1
    r_r2[!is.finite(r_r2)]     <- n_cand + 1
    ranking[idx_cand] <- r_rmse + r_mape + r_r2
  }
  metricas$Ranking <- ranking
  mejor_modelo <- metricas$Modelo[which.min(replace(ranking, !is.finite(ranking), Inf))]
  metricas$EsMejor <- metricas$Modelo == mejor_modelo
  
  # 8.7 Prueba Diebold-Mariano entre RF y XGBoost (errores al cuadrado y absolutos)
  #     y resultado de la función
  dm     <- prueba_dm(real_test - pred_rf, real_test - pred_xgb, perdida = "cuadratica")
  dm_abs <- prueba_dm(real_test - pred_rf, real_test - pred_xgb, perdida = "absoluta")
  
  list(
    objetivo = objetivo, metricas = metricas, metricas_todas = metricas_todas,
    mejor_modelo = mejor_modelo, real_test = real_test, media_obs = media_obs,
    n = c(Entrenamiento = nrow(train), Validacion = nrow(valid), Prueba = nrow(test)),
    origen_hp = origen_hp, dm = dm, dm_abs = dm_abs,
    predicciones = list("Random Forest" = pred_rf, "XGBoost" = pred_xgb),
    modelos = list("Random Forest" = rf2, "XGBoost" = xg2),
    predictores = predictores,
    valores_imputacion = calcular_imputaciones(ajuste, predictores, metodo_na),
    n_predictores = length(predictores))
}

# ============================================================================
# 9. PARTICIÓN EN ENTRENAMIENTO, VALIDACIÓN Y PRUEBA ----
# Aquí se cortan los datos. Los porcentajes llegan desde los deslizadores
# «Entrenamiento» y «Validación» de la barra lateral (sección 14.1).
# ============================================================================
# Reconoce textos como «Entrenamiento/train», «Validación/valid», «Prueba/test».
normalizar_particion <- function(x) {
  x <- tolower(iconv(trimws(as.character(x)), to = "ASCII//TRANSLIT"))
  ifelse(grepl("^(entren|train|trai)", x), "Entrenamiento",
         ifelse(grepl("^(valid|vali)", x), "Validacion",
                ifelse(grepl("^(prueba|test)", x), "Prueba", NA)))
}

## 9.1 Función particionar() ----
# (a) Si el Excel trae una columna «Particion» válida, la respeta tal cual.
# (b) Si no, corta por proporciones: n_tr = floor(n x prop_train),
#     n_va = floor(n x prop_valid), prueba = el resto.
#     Con 547 registros: 382 / 82 / 83. Si la partición es temporal no se
#     baraja: las filas más antiguas van a entrenamiento y las más recientes a prueba.
particionar <- function(df, usar_columna, temporal, prop_train, prop_valid) {
  if (!is.data.frame(df) || nrow(df) < 3) stop("Se necesitan al menos 3 registros.")
  if (usar_columna && "Particion" %in% names(df)) {
    p <- normalizar_particion(df[["Particion"]])
    if (!anyNA(p) && all(c("Entrenamiento", "Prueba") %in% unique(p))) {
      train <- df[p == "Entrenamiento", , drop = FALSE]
      valid <- df[p == "Validacion", , drop = FALSE]
      test  <- df[p == "Prueba", , drop = FALSE]
      if (nrow(train) > 0 && nrow(test) > 0) {
        return(list(train = train, valid = valid, test = test,
                    origen = "columna Particion del archivo"))
      }
    }
  }
  # Corte por proporciones (se usa cuando no hay columna Particion válida)
  n <- nrow(df)
  if (!temporal) { set.seed(2026); df <- df[sample.int(n), , drop = FALSE] }
  n_tr <- max(1, floor(n * prop_train))
  n_va <- max(0, floor(n * prop_valid))
  if (n_tr + n_va >= n) n_va <- max(0, n - n_tr - 1)
  idx_tr <- seq_len(n_tr)
  idx_va <- if (n_va > 0) seq.int(n_tr + 1, n_tr + n_va) else integer(0)
  idx_te <- seq.int(n_tr + n_va + 1, n)
  list(train = df[idx_tr, , drop = FALSE], valid = df[idx_va, , drop = FALSE],
       test = df[idx_te, , drop = FALSE],
       origen = if (temporal) "partición temporal por proporciones" else "partición aleatoria")
}

# ============================================================================
# 10. CONTRASTE DE HIPÓTESIS (Tablas 8 y 9) ----
# Convierte las métricas de prueba en decisiones con la denominación Ho-Hi:
#   Ho-Hi (general), Ho1-Hi1 (recuperación), Ho2-Hi2 (ley), Ho3-Hi3 (consumo)
#   y Ho4-Hi4 (Random Forest vs. XGBoost). Cada hipótesis se decide UNA vez:
#   Ho1-Hi1 a Ho3-Hi3 por variable objetivo; Ho-Hi y Ho4-Hi4 de forma global.
# ============================================================================
## 10.1 Títulos, columnas y notas de las Tablas 8 y 9 ----
# Se usan igual en la pestaña Resultados y en el informe PDF.
TITULO_TABLA8 <- "Tabla 8. Contraste de las hipótesis Ho1-Hi1, Ho2-Hi2 y Ho3-Hi3 en el conjunto de prueba"
TITULO_TABLA9 <- "Tabla 9. Contraste de la hipótesis Ho4-Hi4: Random Forest vs. XGBoost en el conjunto de prueba"
NOTA_TABLA8 <- paste("Nota. Cumple: R² ≥ 0,75, MAPE ≤ 10 % y MAE y RMSE menores que los del modelo base. Se acepta Hi1, Hi2 o Hi3",
                     "si al menos un algoritmo cumple; si ninguno cumple, se acepta Ho. Ho-Hi: se acepta Hi si se aceptan Hi1, Hi2 y Hi3.", sep = "\n")
NOTA_TABLA9 <- paste("Nota. Superioridad significativa: mayor R² y menor MAPE, MAE y RMSE que el otro algoritmo y Diebold-Mariano (HLN)",
                     "con p < 0,05 con errores al cuadrado (e²) y absolutos (|e|). Se acepta Hi4 si ocurre en al menos una variable.", sep = "\n")
COLS_T8 <- c("Variable Objetivo", "Modelo", "Subconjunto", "n", "R²", "MAE", "RMSE", "MAPE (%)", "Cumple criterios", "Decisión")
COLS_T9 <- c("Variable Objetivo", "Modelo", "Subconjunto", "n", "R²", "MAE", "RMSE", "MAPE (%)", "p-valor", "Decisión")
METRICAS_T8 <- c("R²", "MAE", "RMSE", "MAPE (%)")
METRICAS_T9 <- c("R²", "MAE", "RMSE", "MAPE (%)", "p-valor")
FUSION_T8 <- c("Variable Objetivo", "Decisión")             # celdas combinadas por variable
FUSION_T9 <- c("Variable Objetivo", "p-valor", "Decisión")

## 10.2 Denominación y texto de la decisión ----
# Número de la hipótesis específica de cada objetivo (NA si no es objetivo de la tesis).
numero_hipotesis <- function(obj) {
  k <- match(obj, OBJETIVOS_TESIS)
  if (is.na(k)) NA_character_ else as.character(k)
}
# k = "" -> hipótesis general (Ho-Hi); k = "1".."4" -> específicas; NA -> objetivo adicional.
texto_decision <- function(k, acepta) {
  if (is.na(k)) return(if (acepta) "Cumple los criterios" else "No cumple los criterios")
  if (acepta) sprintf("Se acepta Hi%s y se rechaza Ho%s", k, k)
  else sprintf("Se acepta Ho%s y se rechaza Hi%s", k, k)
}
fmt_p <- function(p) if (!is.finite(p)) "—" else if (p < 0.001) "< 0.001" else formatC(p, format = "f", digits = 4)

## 10.3 Función sintesis_hipotesis() ----
# Devuelve una lista con:
#   tabla8  : una fila por objetivo y modelo (modelo base, RF, XGBoost); «Cumple criterios»
#             por modelo y una sola decisión de Hi1/Hi2/Hi3 por variable.
#   tabla9  : una fila por objetivo y modelo (RF, XGBoost) con los p-valores DM y el
#             resultado de la comparación en esa variable.
#   general : Ho-Hi (se acepta Hi si se aceptan Hi1, Hi2 y Hi3).
#   h4      : Ho4-Hi4 (se acepta Hi4 si hay superioridad significativa en al menos una variable).
sintesis_hipotesis <- function(res) {
  t8 <- list(); t9 <- list(); acepta_obj <- logical(0); superior_obj <- logical(0)
  # Orden fijo: recuperación (1), ley (2), consumo (3) y luego objetivos adicionales
  objetivos <- c(intersect(OBJETIVOS_TESIS, res$objetivos), setdiff(res$objetivos, OBJETIVOS_TESIS))
  for (obj in objetivos) {
    r <- res$resultados_por_objetivo[[obj]]; m <- r$metricas
    k <- numero_hipotesis(obj); et <- res$etiquetas_objetivo[[obj]]
    n_test <- length(r$real_test)

    # Tabla 8: ¿qué algoritmos cumplen? -> una decisión para la variable
    cumplen <- m$Modelo[m$Modelo != "Modelo base" & m$Cumple_criterios == "Sí"]
    acepta <- length(cumplen) > 0
    detalle <- if (length(cumplen) == 2) "cumplen ambos algoritmos"
               else if (length(cumplen) == 1) paste("cumple", cumplen) else "no cumple ningún algoritmo"
    decision <- paste0(texto_decision(k, acepta), " (", detalle, ")")
    for (mod in c("Modelo base", "Random Forest", "XGBoost")) {
      f <- m[m$Modelo == mod, , drop = FALSE]; if (nrow(f) == 0) next
      ok <- if (mod == "Modelo base") NA else identical(f$Cumple_criterios, "Sí")
      t8[[length(t8) + 1]] <- data.frame(
        Objetivo = obj, Variable = et, Modelo = mod, Subconjunto = "Prueba", n = n_test,
        R2 = f$R2, MAE = f$MAE, RMSE = f$RMSE, MAPE = f$MAPE,
        Cumple = if (is.na(ok)) "Referencia" else if (ok) "Cumple" else "No cumple",
        Cumple_ok = ok, Decision = decision, Acepta = acepta, stringsAsFactors = FALSE)
    }
    if (!is.na(k)) acepta_obj[obj] <- acepta

    # Tabla 9: Random Forest vs. XGBoost en esta variable
    rf <- m[m$Modelo == "Random Forest", , drop = FALSE]; xg <- m[m$Modelo == "XGBoost", , drop = FALSE]
    if (nrow(rf) == 0 || nrow(xg) == 0) next
    domina <- function(a, b) all(is.finite(c(a$R2, a$MAPE, a$MAE, a$RMSE, b$R2, b$MAPE, b$MAE, b$RMSE))) &&
      a$R2 > b$R2 && a$MAPE < b$MAPE && a$MAE < b$MAE && a$RMSE < b$RMSE
    superior <- if (domina(rf, xg)) "Random Forest" else if (domina(xg, rf)) "XGBoost" else NA_character_
    p_cuad <- r$dm$p_valor; p_abs <- r$dm_abs$p_valor
    sig <- !is.na(superior) && is.finite(p_cuad) && is.finite(p_abs) &&
      p_cuad < UMBRAL_P_TESIS && p_abs < UMBRAL_P_TESIS
    resultado <- if (sig) paste(superior, "significativamente superior") else "Sin superioridad significativa"
    if (!is.na(k)) superior_obj[obj] <- sig
    for (f in list(rf, xg)) {
      t9[[length(t9) + 1]] <- data.frame(
        Objetivo = obj, Variable = et, Modelo = f$Modelo, Subconjunto = "Prueba", n = n_test,
        R2 = f$R2, MAE = f$MAE, RMSE = f$RMSE, MAPE = f$MAPE,
        p_cuad = p_cuad, p_abs = p_abs, Superior = superior,
        Decision = resultado, Acepta = sig, stringsAsFactors = FALSE)
    }
  }
  acepta_g <- length(acepta_obj) > 0 && all(acepta_obj)
  acepta_4 <- length(superior_obj) > 0 && any(superior_obj)
  list(tabla8 = do.call(rbind, t8), tabla9 = do.call(rbind, t9),
       general = list(acepta = acepta_g, decision = texto_decision("", acepta_g),
                      por_objetivo = acepta_obj),
       h4 = list(acepta = acepta_4, decision = texto_decision("4", acepta_4),
                 por_objetivo = superior_obj))
}

## 10.4 Formato de las Tablas 8 y 9 para pantalla y PDF ----
# Devuelven las columnas visibles (COLS_T8 / COLS_T9) como texto, más .grupo
# (variable objetivo, para combinar celdas), .acepta (color de «Decisión») y
# .cumple (color de «Cumple criterios»).
tabla8_mostrar <- function(t) {
  f <- function(x, d) ifelse(is.finite(x), formatC(x, format = "f", digits = d), "—")
  out <- data.frame(t$Variable, t$Modelo, t$Subconjunto, as.character(t$n),
                    f(t$R2, 3), f(t$MAE, 3), f(t$RMSE, 3), f(t$MAPE, 2), t$Cumple, t$Decision,
                    stringsAsFactors = FALSE)
  names(out) <- COLS_T8
  out$.grupo <- t$Objetivo; out$.acepta <- t$Acepta; out$.cumple <- t$Cumple_ok
  out
}
tabla9_mostrar <- function(t) {
  f <- function(x, d) ifelse(is.finite(x), formatC(x, format = "f", digits = d), "—")
  pv <- vapply(seq_len(nrow(t)), function(i)
    paste0("e²: ", fmt_p(t$p_cuad[i]), "\n|e|: ", fmt_p(t$p_abs[i])), character(1))
  out <- data.frame(t$Variable, t$Modelo, t$Subconjunto, as.character(t$n),
                    f(t$R2, 3), f(t$MAE, 3), f(t$RMSE, 3), f(t$MAPE, 2), pv, t$Decision,
                    stringsAsFactors = FALSE)
  names(out) <- COLS_T9
  out$.grupo <- t$Objetivo; out$.acepta <- t$Acepta; out$.cumple <- NA
  out
}

# ============================================================================
# 11. GRÁFICOS (Figuras 1 a 7) ----
# Funciones ggplot2 que se usan en la pestaña Resultados y en el informe PDF.
# ============================================================================
## 11.1 Figura 1: R² por objetivo y modelo ----
# Barras horizontales con la línea roja del umbral 0,75.
grafico_metricas_compacto <- function(res) {
  df <- do.call(rbind, lapply(res$objetivos, function(obj) {
    m <- res$resultados_por_objetivo[[obj]]$metricas
    m <- m[m$Modelo != "Modelo base", , drop = FALSE]
    data.frame(Objetivo = res$etiquetas_objetivo[[obj]], Modelo = m$Modelo,
               R2 = m$R2, MAPE = m$MAPE)
  }))
  df$Modelo <- factor(df$Modelo, levels = c("Random Forest", "XGBoost"))
  ggplot(df, aes(x = R2, y = reorder(Objetivo, R2), fill = Modelo)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.7, alpha = 0.92) +
    geom_vline(xintercept = UMBRAL_R2_TESIS, linetype = "dashed",
               color = "#B91C1C", linewidth = 0.55) +
    geom_text(aes(label = ifelse(is.finite(R2), sprintf("%.3f", R2), "—")),
              position = position_dodge(width = 0.75), hjust = -0.18, size = 3.3,
              color = "#374151") +
    annotate("text", x = UMBRAL_R2_TESIS, y = 0.55, label = "umbral 0,75",
             color = "#B91C1C", size = 3, hjust = -0.05) +
    scale_fill_manual(values = COLORES_MODELO) +
    scale_x_continuous(limits = c(0, 1.12), breaks = seq(0, 1, 0.25)) +
    theme_minimal(base_size = 12) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
          legend.position = "top", legend.title = element_blank(),
          plot.title = element_text(face = "bold", size = 14),
          plot.subtitle = element_text(color = "#6B7280", size = 11),
          axis.title.y = element_blank()) +
    labs(title = "Figura 1. Comparación de las métricas de desempeño en el conjunto de prueba",
         subtitle = "Línea roja = umbral R² = 0,75. La barra más larga gana.",
         x = "R²", y = NULL)
}

## 11.2 Figuras 2, 4 y 6: real vs. predicho ----
# Puntos del conjunto de prueba; la diagonal punteada es la predicción perfecta.
grafico_pred_real <- function(r, et, modelo, num_fig = NULL) {
  pred <- r$predicciones[[modelo]]
  m <- r$metricas[r$metricas$Modelo == modelo, ]
  df_plot <- data.frame(Real = r$real_test, Predicho = pred)
  lim <- range(c(df_plot$Real, df_plot$Predicho))
  titulo <- if (!is.null(num_fig)) paste0("Figura ", num_fig, ". ") else ""
  titulo <- paste0(titulo, "Predicción vs. valor real: ", et)
  ggplot(df_plot, aes(x = Real, y = Predicho)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#9CA3AF") +
    geom_point(alpha = 0.75, color = "#185FA5", size = 2) +
    coord_equal(xlim = lim, ylim = lim) +
    annotate("text", x = lim[1], y = lim[2], hjust = 0, vjust = 1, size = 4,
             label = sprintf("R² = %.3f\nMAE = %.3f\nMAPE = %.2f %%", m$R2, m$MAE, m$MAPE)) +
    theme_minimal(base_size = 13) +
    theme(plot.title = element_text(face = "bold")) +
    labs(title = titulo,
         subtitle = paste0("Modelo: ", modelo, " · prueba (n = ", length(r$real_test), ")"),
         x = paste("Valor real de", et), y = paste("Valor predicho de", et))
}

## 11.3 Figuras 3, 5 y 7: importancia de variables ----
# Random Forest: %IncMSE. XGBoost: Gain. Muestra las 10 variables más importantes.
grafico_importancia_top <- function(modelo, tipo, etiquetas, titulo, top_n = 10, num_fig = NULL) {
  if (tipo == "RF") {
    imp <- as.data.frame(randomForest::importance(modelo))
    df <- data.frame(Variable = rownames(imp), Importancia = imp[, 1])
    ylab <- "%IncMSE"; fill <- COLORES_MODELO[["Random Forest"]]
  } else {
    imp <- as.data.frame(xgboost::xgb.importance(model = modelo))
    df <- data.frame(Variable = imp$Feature, Importancia = imp$Gain)
    ylab <- "Gain"; fill <- COLORES_MODELO[["XGBoost"]]
  }
  df$Variable <- ifelse(df$Variable %in% names(etiquetas), etiquetas[df$Variable], df$Variable)
  df <- df[order(-df$Importancia), ]; df <- utils::head(df, top_n)
  df$Variable <- factor(df$Variable, levels = rev(df$Variable))
  titulo <- if (!is.null(num_fig)) paste0("Figura ", num_fig, ". ", titulo) else titulo
  ggplot(df, aes(x = Variable, y = Importancia)) +
    geom_col(fill = fill, width = 0.7) +
    geom_text(aes(label = ifelse(is.finite(Importancia), sprintf("%.3f", Importancia), "—")),
              hjust = -0.18, size = 3.1, color = "#374151") +
    coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, 0.20))) +
    theme_minimal(base_size = 12) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold", size = 13), axis.title.y = element_blank()) +
    labs(title = titulo, x = NULL, y = ylab)
}

# ============================================================================
# 12. TEMA VISUAL ----
# Colores, fuente y clases CSS de la app (banner, tarjetas, títulos de tablas).
# ============================================================================
tema_app <- bs_theme(
  version = 5, bg = "#FFFFFF", fg = "#111827",
  primary = "#0F6E56", secondary = "#374151",
  base_font = font_google("Inter", local = FALSE),
  heading_font = font_google("Inter", local = FALSE),
  "navbar-bg" = "#FFFFFF"
) |>
  bs_add_rules("
    body { background: #FFFFFF !important; }
    .navbar { border-bottom: 1px solid #E5E7EB !important; box-shadow: none !important; }
    .nav-link { font-weight: 500; color: #6B7280 !important; }
    .nav-link.active { color: #0F6E56 !important; }

    .hero { background: linear-gradient(135deg, #0F6E56 0%, #134E4A 100%);
            color: white; padding: 32px 36px; border-radius: 16px; margin-bottom: 24px; }
    .hero .titulo { font-size: 26px; font-weight: 700; line-height: 1.2; margin: 0 0 8px 0; }
    .hero .subtitulo { font-size: 14px; opacity: 0.9; margin: 0 0 4px 0; }
    .hero .meta { font-size: 12px; opacity: 0.75; margin-top: 12px; }

    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 16px; margin-bottom: 24px; }
    .kpi-card { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
                padding: 20px 22px; box-shadow: 0 2px 12px rgba(0,0,0,0.04); }
    .kpi-card .lbl { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
                     color: #6B7280; font-weight: 600; margin: 0 0 8px 0; }
    .kpi-card .val { font-size: 28px; font-weight: 700; color: #111827; line-height: 1.1; }
    .kpi-card .det { font-size: 12px; color: #6B7280; margin-top: 6px; }

    .verdict { border-radius: 14px; padding: 22px 24px; height: 100%;
               background: #FFFFFF; border: 1px solid #E5E7EB;
               box-shadow: 0 2px 12px rgba(0,0,0,0.05); }
    .verdict.apto { border-left: 6px solid #0F6E56; }
    .verdict.no-apto { border-left: 6px solid #B91C1C; }
    .verdict .titulo { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em;
                       color: #6B7280; margin: 0 0 6px 0; font-weight: 600; }
    .verdict .modelo { font-size: 26px; font-weight: 700; color: #111827; margin: 0; line-height: 1.1; }
    .verdict .badge { display: inline-block; font-size: 11px; font-weight: 700;
                      padding: 3px 10px; border-radius: 999px; margin-top: 8px; }
    .verdict .badge.ok { background: #D1FAE5; color: #065F46; }
    .verdict .badge.no { background: #FEE2E2; color: #991B1B; }
    .verdict .metricas { display: flex; gap: 18px; margin-top: 16px; flex-wrap: wrap; }
    .verdict .metrica .val { font-size: 22px; font-weight: 700; color: #111827; line-height: 1; }
    .verdict .metrica .lbl { font-size: 10px; text-transform: uppercase; color: #9CA3AF;
                             letter-spacing: 0.06em; margin-top: 3px; }
    .verdict .metrica .val.ok { color: #0F6E56; }
    .verdict .metrica .val.no { color: #B91C1C; }

    .hg { border-radius: 14px; padding: 20px 24px; background: #F0FDF4; border: 1px solid #BBF7D0; }
    .hg.no { background: #FEF2F2; border-color: #FECACA; }
    .hg .lbl { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
               color: #065F46; font-weight: 700; }
    .hg.no .lbl { color: #991B1B; }
    .hg .txt { font-size: 17px; font-weight: 600; color: #111827; margin-top: 4px; }

    .protocolo { background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 14px;
                 padding: 18px 22px; font-size: 13px; color: #374151; }
    .protocolo b { color: #111827; }
    .protocolo ul { margin: 8px 0 0 0; padding-left: 20px; }
    .protocolo li { margin-bottom: 4px; }

    .card-seccion { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
                    padding: 22px 26px; box-shadow: 0 2px 12px rgba(0,0,0,0.04);
                    margin-bottom: 18px; }
    .card-seccion h4 { margin-top: 0; color: #0F6E56; font-weight: 700; }
    .card-seccion h5 { color: #111827; font-weight: 600; margin-top: 18px; }
    .card-seccion p, .card-seccion li { color: #374151; font-size: 13.5px; line-height: 1.6; }
    .card-seccion ol { padding-left: 20px; }

    .tabla-titulo { font-size: 14px; font-weight: 700; color: #0F6E56;
                    margin: 18px 0 8px 0; }
    .figura-titulo { font-size: 14px; font-weight: 700; color: #0F6E56;
                     margin: 18px 0 8px 0; }

    .accordion-button { background: #F9FAFB !important; color: #374151 !important;
                        font-weight: 500; border: 1px solid #E5E7EB !important; }
    .accordion-button:not(.collapsed) { background: #FFFFFF !important; color: #0F6E56 !important; }

    table.dataTable thead th { background: #F9FAFB !important; color: #374151 !important;
                               border-bottom: 1px solid #E5E7EB !important;
                               font-weight: 600 !important; font-size: 12px; }
    table.dataTable tbody td { border-bottom: 1px solid #F3F4F6 !important; font-size: 13px; }

    /* Tablas 8 y 9: contraste de hipótesis con celdas combinadas */
    .tabla-hip { width: 100%; border-collapse: collapse; font-size: 13px; background: #FFFFFF; }
    .tabla-hip th { background: #0F6E56; color: #FFFFFF; font-weight: 600; text-align: center;
                    vertical-align: middle; padding: 6px 8px; border: 1px solid #D1D5DB; }
    .tabla-hip td { padding: 7px 8px; border: 1px solid #D1D5DB; text-align: center;
                    vertical-align: middle; color: #111827; }
    .tabla-hip td.var { font-weight: 600; text-align: center; background: #F9FAFB; }
    .tabla-hip td.dec-ok { background: #E1F5EE; color: #065F46; font-weight: 600; }
    .tabla-hip td.dec-no { background: #FEE2E2; color: #991B1B; font-weight: 600; }
    .tabla-hip td.dec-ref { background: #F3F4F6; color: #6B7280; }
    .tabla-hip td.pval { white-space: pre; min-width: 120px; }
    .nota-hip { font-size: 12px; color: #4B5563; margin-top: 6px; white-space: pre-line; }
  ")

# ============================================================================
# 13. COMPONENTES DE INTERFAZ ----
# Piezas HTML reutilizables: formato de números, tarjetas KPI y de veredicto.
# ============================================================================
fmt_num <- function(x, d = 3) {
  if (!is.finite(x)) return("—")
  formatC(x, format = "f", digits = d, big.mark = " ")
}
clase_ok <- function(cond) if (isTRUE(cond)) "ok" else "no"

# Tarjeta de indicador (Portada: R² y MAPE del mejor modelo).
kpi_card <- function(titulo, valor, detalle = NULL) {
  tags$div(class = "kpi-card",
           tags$p(class = "lbl", titulo),
           tags$div(class = "val", valor),
           if (!is.null(detalle)) tags$div(class = "det", detalle))
}

# Tarjeta APTO / NO APTO por objetivo; colorea cada métrica según su umbral.
tarjeta_veredicto <- function(etiqueta_obj, mejor_modelo, fila, cumple) {
  apto <- identical(cumple, "Sí")
  clase <- if (apto) "apto" else "no-apto"
  badge <- if (apto) "APTO" else "NO APTO"
  clase_badge <- if (apto) "ok" else "no"
  r2_ok    <- is.finite(fila$R2) && fila$R2 >= UMBRAL_R2_TESIS
  mape_ok  <- is.finite(fila$MAPE) && fila$MAPE <= UMBRAL_MAPE_TESIS
  red_ok   <- is.finite(fila$Red_MAE_vs_base) && fila$Red_MAE_vs_base > 0 &&
    is.finite(fila$Red_RMSE_vs_base) && fila$Red_RMSE_vs_base > 0
  tags$div(class = paste("verdict", clase),
           tags$p(class = "titulo", etiqueta_obj),
           tags$p(class = "modelo", mejor_modelo),
           tags$span(class = paste("badge", clase_badge), badge),
           tags$div(class = "metricas",
                    tags$div(class = "metrica",
                             tags$div(class = paste("val", clase_ok(r2_ok)), fmt_num(fila$R2, 3)),
                             tags$div(class = "lbl", "R²")),
                    tags$div(class = "metrica",
                             tags$div(class = paste("val", clase_ok(mape_ok)), paste0(fmt_num(fila$MAPE, 2), " %")),
                             tags$div(class = "lbl", "MAPE")),
                    tags$div(class = "metrica",
                             tags$div(class = paste("val", clase_ok(red_ok)), paste0(fmt_num(fila$Red_MAE_vs_base, 0), " %")),
                             tags$div(class = "lbl", "Red. MAE")),
                    tags$div(class = "metrica",
                             tags$div(class = paste("val", clase_ok(red_ok)), paste0(fmt_num(fila$Red_RMSE_vs_base, 0), " %")),
                             tags$div(class = "lbl", "Red. RMSE"))))
}

# Recuadro verde o rojo con la decisión de la hipótesis general Ho-Hi.
tarjeta_hg <- function(sintesis) {
  hg <- sintesis$general
  tags$div(class = paste("hg", if (!hg$acepta) "no"),
           tags$div(class = "lbl", "Hipótesis general (Ho-Hi)"),
           tags$div(class = "txt", hg$decision),
           tags$div(style = "font-size: 12px; color: #6B7280; margin-top: 6px;",
                    "Criterios: R² ≥ 0,75 · MAPE ≤ 10 % · MAE y RMSE menores que los del modelo base."))
}

## 13.1 Tablas de contraste de hipótesis (Tablas 8 y 9) ----
# t        : salida de tabla8_mostrar() o tabla9_mostrar() (sección 10.4).
# columnas : columnas visibles; metricas: columnas bajo «Métricas de Evaluación de Modelo».
# fusion   : columnas cuyas celdas se combinan dentro de cada variable objetivo.
# «Decisión» se colorea con .acepta y «Cumple criterios» con .cumple:
# verde = sí, rojo = no, gris = modelo de referencia.

# Versión HTML (pestaña Resultados).
tabla_hip_html <- function(t, columnas, metricas, fusion) {
  i_m <- match(metricas, columnas)
  antes <- columnas[seq_len(min(i_m) - 1)]
  despues <- columnas[-seq_len(max(i_m))]
  clase_dec <- function(a) if (is.na(a)) "dec-ref" else if (isTRUE(a)) "dec-ok" else "dec-no"
  cab1 <- tags$tr(lapply(antes, function(c) tags$th(rowspan = 2, c)),
                  tags$th(colspan = length(metricas), "Métricas de Evaluación de Modelo"),
                  lapply(despues, function(c) tags$th(rowspan = 2, c)))
  cab2 <- tags$tr(lapply(metricas, function(c) tags$th(c)))
  filas <- list()
  for (g in unique(t$.grupo)) {
    sub <- t[t$.grupo == g, , drop = FALSE]; n <- nrow(sub)
    for (i in seq_len(n)) {
      celdas <- lapply(columnas, function(c) {
        clase <- c(if (c == "Variable Objetivo") "var",
                   if (c == "Decisión") clase_dec(sub$.acepta[i]),
                   if (c == "Cumple criterios") clase_dec(sub$.cumple[i]),
                   if (c == "p-valor") "pval")
        if (c %in% fusion) {
          if (i > 1) return(NULL)
          return(tags$td(rowspan = n, class = paste(clase, collapse = " "), sub[[c]][1]))
        }
        tags$td(class = paste(clase, collapse = " "), sub[[c]][i])
      })
      filas[[length(filas) + 1]] <- tags$tr(Filter(Negate(is.null), celdas))
    }
  }
  tags$table(class = "tabla-hip", tags$thead(cab1, cab2), tags$tbody(filas))
}

# Versión gráfica (informe PDF): gtable con celdas combinadas; anchos en cm.
tabla_hip_grob <- function(t, columnas, metricas, fusion, anchos, tam = 8.5) {
  VERDE <- "#0F6E56"
  i_m <- match(metricas, columnas)
  envolver_celda <- function(x, ancho_cm) {
    max_car <- max(6, floor(ancho_cm / (tam * 0.0205)))
    paste(unlist(lapply(strsplit(as.character(x), "\n", fixed = TRUE)[[1]],
                        function(s) strwrap(s, width = max_car))), collapse = "\n")
  }
  lineas <- function(x) length(strsplit(x, "\n", fixed = TRUE)[[1]])
  txt <- t
  for (j in seq_along(columnas)) txt[[columnas[j]]] <- vapply(t[[columnas[j]]], envolver_celda,
                                                              character(1), ancho_cm = anchos[j])
  alto_linea <- tam * 0.045; margen <- 0.3        # cm
  # Altura de cada fila: la mayor de sus celdas sin combinar; luego se ajusta para
  # que las celdas combinadas quepan en su grupo.
  alturas <- vapply(seq_len(nrow(txt)), function(i)
    max(1, vapply(setdiff(columnas, fusion), function(c) lineas(txt[[c]][i]), numeric(1))) * alto_linea + margen,
    numeric(1))
  for (g in unique(txt$.grupo)) {
    idx <- which(txt$.grupo == g)
    necesario <- max(vapply(fusion, function(c) lineas(txt[[c]][idx[1]]), numeric(1))) * alto_linea + margen
    if (sum(alturas[idx]) < necesario) alturas[idx] <- alturas[idx] * necesario / sum(alturas[idx])
  }
  g <- gtable::gtable(widths = grid::unit(anchos, "cm"),
                      heights = grid::unit(c(0.6, 0.6, alturas), "cm"))
  contador <- 0
  celda <- function(g, etiqueta, t_, b_, l_, r_, fondo, col_txt = "#111827", negrita = FALSE) {
    contador <<- contador + 1   # nombres únicos: gtable descarta grobs con nombre repetido
    g <- gtable::gtable_add_grob(g, grid::rectGrob(gp = grid::gpar(fill = fondo, col = "grey70", lwd = 0.6)),
                                 t = t_, b = b_, l = l_, r = r_, z = 1, clip = "off",
                                 name = paste0("fondo", contador))
    gtable::gtable_add_grob(g, grid::textGrob(etiqueta, gp = grid::gpar(fontsize = tam, col = col_txt,
                                                                       fontface = if (negrita) "bold" else "plain")),
                            t = t_, b = b_, l = l_, r = r_, z = 2, clip = "off",
                            name = paste0("texto", contador))
  }
  # Encabezado en dos niveles
  for (j in seq_along(columnas)) {
    if (j %in% i_m) g <- celda(g, columnas[j], 2, 2, j, j, VERDE, "white", TRUE)
    else g <- celda(g, columnas[j], 1, 2, j, j, VERDE, "white", TRUE)
  }
  g <- celda(g, "Métricas de Evaluación de Modelo", 1, 1, min(i_m), max(i_m), VERDE, "white", TRUE)
  # Cuerpo
  fondo_dec <- function(a) if (is.na(a)) "#F3F4F6" else if (isTRUE(a)) "#E1F5EE" else "#FEE2E2"
  color_dec <- function(a) if (is.na(a)) "#6B7280" else if (isTRUE(a)) "#065F46" else "#991B1B"
  for (gr in unique(txt$.grupo)) {
    idx <- which(txt$.grupo == gr)
    for (i in idx) for (j in seq_along(columnas)) {
      c <- columnas[j]
      es_dec <- c %in% c("Decisión", "Cumple criterios")
      estado <- if (c == "Cumple criterios") txt$.cumple[i] else txt$.acepta[i]
      fondo <- if (es_dec) fondo_dec(estado) else if (c == "Variable Objetivo") "#F9FAFB" else "white"
      colt <- if (es_dec) color_dec(estado) else "#111827"
      if (c %in% fusion) {
        if (i != idx[1]) next
        g <- celda(g, txt[[c]][i], 2 + idx[1], 2 + max(idx), j, j, fondo, colt, es_dec || c == "Variable Objetivo")
      } else {
        g <- celda(g, txt[[c]][i], 2 + i, 2 + i, j, j, fondo, colt, es_dec)
      }
    }
  }
  g
}



# ============================================================================
# 14. INTERFAZ DE USUARIO (UI) ----
# Solo declara controles y espacios de salida; no calcula nada. Cada salida
# tiene un ID (ej. DTOutput("tabla4")) que el servidor rellena (output$tabla4).
# ============================================================================
ui <- page_sidebar(
  title = NULL, theme = tema_app, fillable = FALSE,
  ## 14.1 Barra lateral ----
  # Carga de datos, selección de variables, opciones avanzadas, botón y descarga.
  sidebar = sidebar(
    width = 340, open = "open",
    tags$div(style = "padding: 4px 0 12px 0; border-bottom: 1px solid #E5E7EB; margin-bottom: 14px;",
             tags$div(style = "font-size: 15px; font-weight: 700; color: #0F6E56;",
                      icon("chart-line"), " Flotación Au · 2026"),
             tags$div(style = "font-size: 11px; color: #6B7280; margin-top: 2px;",
                      "Barboza & Ruiz · UCV · Maestría IA")),
    
    # Carga del archivo de datos (Excel o CSV)
    fileInput("archivo", NULL, buttonLabel = "Cargar dataset",
              placeholder = "CSV o Excel", accept = c(".xlsx", ".xls", ".csv")),
    uiOutput("aviso_carga"),
    
    tags$div(style = "margin-top: 14px;",
             # Selectores de variables (se construyen en el servidor, sección 15.2)
             uiOutput("selector_objetivos"),
             uiOutput("selector_predictores")),
    
    accordion(open = FALSE,
              # Opciones avanzadas:
              #   usar_columna_particion : usa la columna «Particion» del Excel si existe
              #   prop_train / prop_valid: PORCENTAJES DE LA PARTICIÓN (defecto 0.70 y 0.15;
              #                            la prueba es el resto). Se aplican en particionar().
              #   particion_temporal     : mantiene el orden cronológico (no baraja)
              #   manejo_na              : imputar faltantes con la media o con 0
              #   usar_hp_tesis          : usa HP_TESIS (sección 2.3) o hiperparámetros genéricos
              accordion_panel("Opciones avanzadas", icon = icon("sliders"),
                              checkboxInput("usar_columna_particion", "Usar columna 'Particion'", value = TRUE),
                              sliderInput("prop_train", "Entrenamiento", 0.5, 0.9, 0.7, 0.05),
                              sliderInput("prop_valid", "Validación", 0.05, 0.3, 0.15, 0.05),
                              checkboxInput("particion_temporal", "Partición temporal (orden cronológico)", value = TRUE),
                              radioButtons("manejo_na", "Faltantes",
                                           c("Imputar con media de ajuste" = "media", "Rellenar con 0" = "cero"),
                                           selected = "media"),
                              checkboxInput("usar_hp_tesis", "Hiperparámetros de la tesis", value = TRUE))),
    
    tags$div(style = "margin-top: 16px;",
             # Botón que dispara todo el cálculo (resultado_modelos, sección 15.4)
             actionButton("entrenar", tagList(icon("play"), "Ejecutar protocolo"),
                          class = "btn-primary w-100",
                          style = "font-weight:600; padding:10px;")),
    br(),
    # Botón de descarga del informe PDF (aparece tras ejecutar)
    uiOutput("botones_descarga")
  ),
  
  ## 14.2 Pestañas ----
  # Cada nav_panel es una pestaña; sus salidas se rellenan en la sección 15.
  navset_card_tab(
    nav_panel(
      # Pestaña PORTADA (servidor: sección 15.5)
      tagList(icon("house"), " Portada"), br(),
      uiOutput("hero"),
      uiOutput("kpis_abstract"),
      uiOutput("protocolo_card"),
      br(),
      uiOutput("hg_card"),
      br(),
      uiOutput("tarjetas_veredicto")
    ),
    
    nav_panel(
      # Pestaña DATOS (servidor: sección 15.6)
      tagList(icon("database"), " Datos"), br(),
      uiOutput("resumen_particion"),
      br(),
      uiOutput("selector_sub_desc"),
      DTOutput("tabla_descriptivos"),
      br(),
      plotOutput("grafico_boxplot_sub", height = "340px"),
      br(),
      DTOutput("tabla_datos")
    ),
    
    nav_panel(
      # Pestaña RESULTADOS: Tablas 4-9 y Figuras 1-7 en el orden del informe (servidor: 15.7 y 15.8)
      tagList(icon("chart-simple"), " Resultados"), br(),
      tags$div(class = "tabla-titulo", "Tabla 4. Desempeño predictivo de los modelos en el conjunto de prueba"),
      DTOutput("tabla4"),
      br(),
      tags$div(class = "figura-titulo", "Figura 1. Comparación de las métricas de desempeño"),
      plotOutput("figura1", height = "500px"),
      br(),
      # Recuperación de Au (Ho1-Hi1)
      tags$div(class = "tabla-titulo", "Tabla 5. Recuperación de Au (%) por subconjunto"),
      DTOutput("tabla5"),
      tags$div(class = "figura-titulo", "Figura 2. Valores observados vs. predichos de la recuperación de Au"),
      plotOutput("figura2", height = "420px"),
      tags$div(class = "figura-titulo", "Figura 3. Importancia de variables en recuperación de Au"),
      fluidRow(column(6, plotOutput("figura3_rf", height = "360px")),
               column(6, plotOutput("figura3_xgb", height = "360px"))),
      br(),
      # Ley de Au en el concentrado (Ho2-Hi2)
      tags$div(class = "tabla-titulo", "Tabla 6. Ley de Au en el concentrado (g/t) por subconjunto"),
      DTOutput("tabla6"),
      tags$div(class = "figura-titulo", "Figura 4. Valores observados vs. predichos de la ley de Au"),
      plotOutput("figura4", height = "420px"),
      tags$div(class = "figura-titulo", "Figura 5. Importancia de variables en ley de Au"),
      fluidRow(column(6, plotOutput("figura5_rf", height = "360px")),
               column(6, plotOutput("figura5_xgb", height = "360px"))),
      br(),
      # Consumo específico de reactivos (Ho3-Hi3)
      tags$div(class = "tabla-titulo", "Tabla 7. Consumo específico de reactivos (kg/t) por subconjunto"),
      DTOutput("tabla7"),
      tags$div(class = "figura-titulo", "Figura 6. Valores observados vs. predichos del consumo"),
      plotOutput("figura6", height = "420px"),
      tags$div(class = "figura-titulo", "Figura 7. Importancia de variables en consumo de reactivos"),
      fluidRow(column(6, plotOutput("figura7_rf", height = "360px")),
               column(6, plotOutput("figura7_xgb", height = "360px"))),
      br(),
      # Contraste de hipótesis (servidor: sección 15.7; funciones en secciones 10 y 13)
      tags$div(class = "tabla-titulo", TITULO_TABLA8),
      uiOutput("tabla8"),
      br(),
      tags$div(class = "tabla-titulo", TITULO_TABLA9),
      uiOutput("tabla9")
    ),
    
    # Pestañas DISCUSIÓN y CONCLUSIONES: texto fijo del informe (servidor: 15.9)
    nav_panel(tagList(icon("comments"), " Discusión"), br(), uiOutput("discusion_card")),
    
    nav_panel(tagList(icon("list-check"), " Conclusiones"), br(),
              uiOutput("conclusiones_card"), br(), uiOutput("recomendaciones_card"))

  )
)

# ============================================================================
# 15. SERVIDOR (lógica reactiva) ----
# Lee los controles, ejecuta el protocolo al pulsar «Ejecutar protocolo» y
# rellena cada salida de la interfaz.
# ============================================================================
server <- function(input, output, session) {
  
  ## 15.1 Carga de datos ----
  # datos_cargados() lee el Excel/CSV subido con «Cargar dataset» (sección 3).
  # Mientras no se cargue un archivo devuelve NULL y la app espera. aviso_carga
  # muestra el mensaje verde/rojo bajo el botón «Cargar dataset».
  resultado_carga <- reactiveVal(NULL)
  
  datos_cargados <- reactive({
    if (!is.null(input$archivo)) {
      df <- tryCatch(leer_archivo_robusto(input$archivo$datapath, input$archivo$name),
                     error = function(e) {
                       resultado_carga(paste("No se pudo leer el archivo:", conditionMessage(e)))
                       NULL })
      if (!is.null(df)) {
        msg <- sprintf("Archivo cargado: %d registros × %d columnas.", nrow(df), ncol(df))
        if ("Particion" %in% names(df)) {
          tab <- table(normalizar_particion(df[["Particion"]]), useNA = "no")
          msg <- paste0(msg, " Partición: ", paste(names(tab), tab, sep = " = ", collapse = ", "), ".")
        }
        resultado_carga(msg); df <- sanear_nombres(df)
      }
      df
    } else NULL
  })
  
  output$aviso_carga <- renderUI({
    msg <- resultado_carga(); req(msg)
    color <- if (grepl("^No se pudo", msg)) "#B91C1C" else "#0F6E56"
    tags$div(style = paste0("font-size:12px; color:", color, "; margin-top:4px;"), msg)
  })
  
  ## 15.2 Selectores de variables ----
  # columnas_numericas(): candidatas a objetivo/predictor (excluye año, mes, id...).
  # pct_faltante(): % de faltantes por columna, que se muestra junto al nombre.
  # Por defecto se marcan los 3 objetivos de la tesis y los predictores con <= 50 % de faltantes.
  columnas_numericas <- reactive({
    df <- datos_cargados(); req(df)
    num <- names(df)[sapply(df, is.numeric)]
    num[!grepl(PATRON_META, tolower(num))]
  })
  pct_faltante <- reactive({
    df <- datos_cargados(); cols <- columnas_numericas(); req(df, cols)
    sapply(cols, function(c) mean(is.na(df[[c]])))
  })
  etiquetas_con_faltante <- function(df, cols, pf) {
    vapply(cols, function(c) {
      base <- etiqueta(df, c)
      if (!is.na(pf[[c]]) && pf[[c]] > 0) paste0(base, "  (", round(100 * pf[[c]]), "% falt.)") else base
    }, character(1))
  }
  
  output$selector_objetivos <- renderUI({
    df <- datos_cargados(); cols <- columnas_numericas(); req(df, cols)
    objetivos_tesis <- intersect(OBJETIVOS_TESIS, cols)
    seleccionados <- if (length(objetivos_tesis) > 0) objetivos_tesis else cols[1]
    pf <- pct_faltante()
    selectizeInput("objetivos", "Variables objetivo",
                   choices = stats::setNames(cols, etiquetas_con_faltante(df, cols, pf)),
                   selected = seleccionados, multiple = TRUE,
                   options = list(plugins = list("remove_button"),
                                  placeholder = "Busca y selecciona..."))
  })
  
  output$selector_predictores <- renderUI({
    df <- datos_cargados(); cols <- columnas_numericas(); req(df, cols, input$objetivos)
    disponibles <- setdiff(cols, input$objetivos)
    req(length(disponibles) > 0)
    pf <- pct_faltante()
    seleccionados_default <- disponibles[vapply(disponibles, function(c) isTRUE(pf[[c]] <= 0.5), logical(1))]
    selectizeInput("predictores", "Variables predictoras",
                   choices = stats::setNames(disponibles, etiquetas_con_faltante(df, disponibles, pf)),
                   selected = seleccionados_default, multiple = TRUE,
                   options = list(plugins = list("remove_button"),
                                  placeholder = "Busca y selecciona..."))
  })
  
  ## 15.3 Predictores efectivos por objetivo ----
  # Predictores efectivos por objetivo:
  #   - Las dosificaciones de los 5 reactivos (REACTIVOS_TESIS) entran siempre,
  #     para los tres objetivos, aunque no se hayan marcado en el selector.
  #   - El consumo específico total entra como predictor de recuperación y ley.
  #   - Un objetivo nunca se usa para predecirse a sí mismo.
  predictores_para <- function(objetivo, predictores, columnas) {
    p <- union(predictores, intersect(REACTIVOS_TESIS, columnas))
    if ("consumo_reactivos" %in% columnas) p <- union(p, "consumo_reactivos")
    setdiff(p, objetivo)
  }
  
  # --------------------------------------------------------------------
  ## 15.4 Motor del protocolo: resultado_modelos ----
  # Se ejecuta SOLO al pulsar «Ejecutar protocolo». Pasos A a F:
  # --------------------------------------------------------------------
  resultado_modelos <- eventReactive(input$entrenar, {
    df <- datos_cargados()
    validate(
      need(df, "Carga un dataset (Excel o CSV) con el botón «Cargar dataset»."),
      need(length(input$objetivos) >= 1, "Selecciona al menos una variable objetivo."),
      need(length(input$predictores) >= 1, "Selecciona al menos una variable predictora."))
    
    # A. Si no se usa la columna Particion, ordena por Fecha y Guardia (orden cronológico)
    usar_columna <- isTRUE(input$usar_columna_particion) && "Particion" %in% names(df)
    if (!usar_columna && "Fecha" %in% names(df)) {
      fechas <- suppressWarnings(as.Date(as.character(df[["Fecha"]]),
                                         tryFormats = c("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y")))
      if (!all(is.na(fechas))) {
        orden_secundario <- if ("Guardia" %in% names(df)) df[["Guardia"]] else seq_len(nrow(df))
        df <- df[order(fechas, orden_secundario), ]
      }
    }
    # B. PARTICIÓN: valida los porcentajes de la barra lateral y llama a particionar()
    #    (sección 9). Exige >= 30 registros de entrenamiento y >= 5 de prueba.
    validate(need(input$prop_train + input$prop_valid < 0.95,
                  "La suma de proporciones debe ser menor que 0,95."))
    part <- particionar(df, usar_columna, isTRUE(input$particion_temporal),
                        input$prop_train, input$prop_valid)
    validate(need(nrow(part$train) >= 30 && nrow(part$test) >= 5,
                  "Se requieren ≥30 registros de entrenamiento y ≥5 de prueba."))
    
    # C. Hiperparámetros genéricos (solo si se desmarca «Hiperparámetros de la tesis»)
    hp_ui <- list(
      rf  = list(ntree = 300, nodesize = 5, max_features = 0.6, max_depth = NA),
      xgb = list(nrounds = 200, eta = 0.05, max_depth = 3, subsample = 0.8, colsample = 0.8))
    
    # D. Entrena y evalúa cada objetivo con entrenar_evaluar_target() (sección 8), con barra de progreso
    resultados_por_objetivo <- withProgress(message = "Ejecutando protocolo...", value = 0, {
      lapply(seq_along(input$objetivos), function(i) {
        obj <- input$objetivos[i]
        incProgress(1 / length(input$objetivos), detail = etiqueta(df, obj))
        cols_num <- names(df)[vapply(df, is.numeric, logical(1))]
        preds <- predictores_para(obj, input$predictores, cols_num)
        validate(need(length(preds) >= 1, paste("Sin predictores para", etiqueta(df, obj))))
        validate(need(obj %in% names(df), paste("No se encuentra:", obj)))
        validate(need(is.numeric(df[[obj]]), paste("La variable objetivo debe ser numérica:", obj)))
        validate(need(sum(is.finite(df[[obj]])) >= 10, paste("Hay <10 valores válidos para:", obj)))
        entrenar_evaluar_target(obj, preds, part$train, part$valid, part$test, hp_ui,
                                isTRUE(input$usar_hp_tesis), input$manejo_na)
      })
    })
    names(resultados_por_objetivo) <- input$objetivos
    
    # E. Etiquetas de los objetivos, descriptivos y datos particionados (pestaña Datos)
    etiquetas_objetivo <- stats::setNames(etiquetas_de(df, input$objetivos), input$objetivos)
    
    vars_desc <- unique(c(input$objetivos, input$predictores))
    descriptivos <- descriptivos_por_subconjunto(part$train, part$valid, part$test, vars_desc)
    datos_particionados <- rbind(
      cbind(Subconjunto = "Entrenamiento", part$train[, vars_desc, drop = FALSE]),
      cbind(Subconjunto = "Validación",    part$valid[, vars_desc, drop = FALSE]),
      cbind(Subconjunto = "Prueba",        part$test [, vars_desc, drop = FALSE]))
    
    # F. Contraste de hipótesis (Tablas 8 y 9, sección 10) y lista de resultados que leen todas las salidas
    sintesis <- sintesis_hipotesis(list(
      objetivos = input$objetivos, etiquetas_objetivo = etiquetas_objetivo,
      resultados_por_objetivo = resultados_por_objetivo))
    
    list(
      resultados_por_objetivo = resultados_por_objetivo,
      objetivos = input$objetivos, etiquetas_objetivo = etiquetas_objetivo,
      etiquetas_datos = attr(df, "etiquetas"),
      particion = list(origen = part$origen, n_train = nrow(part$train),
                       n_valid = nrow(part$valid), n_test = nrow(part$test)),
      descriptivos = descriptivos, datos_particionados = datos_particionados,
      sintesis = sintesis)
  })
  
  # --------------------------------------------------------------------
  ## 15.5 Pestaña Portada ----
  # Banner, KPIs, protocolo, hipótesis general y tarjetas de veredicto. Antes de
  # ejecutar muestra contenido por defecto (tryCatch devuelve NULL).
  # --------------------------------------------------------------------
  output$hero <- renderUI({
    tags$div(class = "hero",
             tags$p(class = "subtitulo", "Trabajo de investigación para obtener el grado de maestro en Inteligencia Artificial"),
             tags$h1(class = "titulo", "Aplicación de machine learning para predecir la flotación de relaves de lixiviación, Lima 2026"),
             tags$p(class = "subtitulo", "Barboza Duran, Edwin Miguel · Ruiz Torres, Alfonso"),
             tags$p(class = "subtitulo", "Asesor: MSc. Ruben Quispe Llacctarimay"),
             tags$p(class = "meta", "Universidad César Vallejo · Escuela de Posgrado · Maestría en Inteligencia Artificial"),
             tags$p(class = "meta", "Línea: Machine Learning para gestión de decisiones empresariales · ODS 9 – Industria, Innovación e Infraestructura"))
  })
  
  output$kpis_abstract <- renderUI({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL)
    if (is.null(res) || is.null(res$resultados_por_objetivo) ||
        length(res$resultados_por_objetivo) == 0) {
      return(tags$div(class = "kpi-grid",
                      kpi_card("Dataset", "547 registros", "38 predictores · enero–setiembre 2026"),
                      kpi_card("Partición", "70 / 15 / 15", "Entrenamiento · Validación · Prueba (temporal)"),
                      kpi_card("Criterios", "R² ≥ 0.75", "MAPE ≤ 10 % · MAE y RMSE < modelo base"),
                      kpi_card("Prueba estadística", "DM-HLN", "Diebold-Mariano con corrección HLN")))
    }
    tarjetas <- lapply(res$objetivos, function(obj) {
      r <- res$resultados_por_objetivo[[obj]]
      if (is.null(r)) return(NULL)
      m <- r$metricas[r$metricas$Modelo == r$mejor_modelo, , drop = FALSE]
      if (nrow(m) == 0) return(NULL)
      kpi_card(res$etiquetas_objetivo[[obj]],
               sprintf("R² = %.3f", m$R2[1]),
               paste0(r$mejor_modelo, " · MAPE = ", fmt_num(m$MAPE[1], 2), " %"))
    })
    tarjetas <- Filter(Negate(is.null), tarjetas)
    tags$div(class = "kpi-grid", tarjetas)
  })
  
  output$protocolo_card <- renderUI({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL)
    p <- if (!is.null(res) && !is.null(res$particion)) res$particion
    else list(origen = "—", n_train = "—", n_valid = "—", n_test = "—")
    tags$div(class = "protocolo",
             tags$b("Protocolo de la tesis"),
             tags$ul(
               tags$li(sprintf("Partición: %s — entrenamiento = %s, validación = %s, prueba = %s registros.",
                               p$origen, p$n_train, p$n_valid, p$n_test)),
               tags$li("Modelo base: media histórica del conjunto de ajuste (entrenamiento + validación)."),
               tags$li("Entrenamiento y validación: modelos ajustados solo con entrenamiento."),
               tags$li("Prueba: modelos reentrenados con entrenamiento + validación, evaluados una única vez."),
               tags$li("Criterios Ho-Hi y Ho1-Hi1 a Ho3-Hi3: R² ≥ 0,75; MAPE ≤ 10 %; MAE y RMSE menores que los del modelo base. Ho4-Hi4: mayor R² y menor MAPE, MAE y RMSE que el otro modelo, y DM con p < 0,05 con errores al cuadrado y absolutos."),
               tags$li("Las dosificaciones de los cinco reactivos se usan como predictores de las tres variables; el consumo específico total se usa como predictor de la recuperación y de la ley.")))
  })
  
  output$hg_card <- renderUI({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL)
    if (is.null(res) || is.null(res$sintesis)) return(NULL)
    tarjeta_hg(res$sintesis)
  })
  
  output$tarjetas_veredicto <- renderUI({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL)
    if (is.null(res) || is.null(res$resultados_por_objetivo)) return(NULL)
    tarjetas <- lapply(res$objetivos, function(obj) {
      r <- res$resultados_por_objetivo[[obj]]
      if (is.null(r)) return(NULL)
      fila <- r$metricas[r$metricas$Modelo == r$mejor_modelo, , drop = FALSE]
      if (nrow(fila) == 0) return(NULL)
      column(4, tarjeta_veredicto(res$etiquetas_objetivo[[obj]], r$mejor_modelo,
                                  fila, fila$Cumple_criterios))
    })
    tarjetas <- Filter(Negate(is.null), tarjetas)
    if (length(tarjetas) == 0) return(NULL)
    do.call(fluidRow, tarjetas)
  })
  
  # --------------------------------------------------------------------
  ## 15.6 Pestaña Datos ----
  # Tamaño de cada subconjunto, descriptivos y boxplot de la variable elegida, y
  # tabla del dataset cargado.
  # --------------------------------------------------------------------
  output$resumen_particion <- renderUI({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL)
    if (is.null(res) || is.null(res$particion)) return(NULL)
    p <- res$particion
    mini_card <- function(titulo, valor, detalle, color = "#0F6E56") {
      tags$div(style = paste0("background:#FFFFFF; border:1px solid #E5E7EB; border-left:5px solid ",
                              color, "; border-radius:10px; padding:16px 20px; height:100%;"),
               tags$div(style = "font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
                          color:#6B7280; font-weight:600;", titulo),
               tags$div(style = "font-size:22px; font-weight:700; color:#111827; margin-top:4px;", valor),
               tags$div(style = "font-size:12px; color:#6B7280; margin-top:4px;", detalle))
    }
    fluidRow(
      column(3, mini_card("Entrenamiento", p$n_train, "registros")),
      column(3, mini_card("Validación", p$n_valid, "registros")),
      column(3, mini_card("Prueba", p$n_test, "registros")),
      column(3, mini_card("Origen", "—", p$origen, color = "#185FA5")))
  })
  
  output$selector_sub_desc <- renderUI({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL)
    if (is.null(res) || is.null(res$descriptivos)) return(NULL)
    opciones <- unique(res$descriptivos$Variable)
    selectInput("var_desc", "Variable a describir por subconjunto",
                choices = stats::setNames(opciones, opciones))
  })
  
  output$tabla_descriptivos <- renderDT({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL)
    req(res, input$var_desc)
    d <- res$descriptivos; d <- d[d$Variable == input$var_desc, ]
    et <- res$etiquetas_datos
    d$Variable <- if (!is.null(et) && input$var_desc %in% names(et)) unname(et[[input$var_desc]]) else input$var_desc
    num <- c("Media","DE","Min","Q1","Mediana","Q3","Max","CV_pct")
    d[num] <- lapply(d[num], round, 4)
    datatable(d, rownames = FALSE, options = list(dom = "t", pageLength = -1, scrollX = TRUE))
  })
  
  output$grafico_boxplot_sub <- renderPlot({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL)
    req(res, input$var_desc)
    d <- res$datos_particionados
    d$Subconjunto <- factor(d$Subconjunto, levels = c("Entrenamiento","Validación","Prueba"))
    et <- res$etiquetas_datos
    etq <- if (!is.null(et) && input$var_desc %in% names(et)) unname(et[[input$var_desc]]) else input$var_desc
    ggplot(d, aes(x = Subconjunto, y = .data[[input$var_desc]], fill = Subconjunto)) +
      geom_boxplot(alpha = 0.85, na.rm = TRUE) +
      geom_jitter(width = 0.15, alpha = 0.25, size = 1, na.rm = TRUE) +
      scale_fill_manual(values = c("Entrenamiento" = "#2A78D6",
                                   "Validación" = "#EB6834", "Prueba" = "#1BAF7A")) +
      theme_minimal(base_size = 13) +
      labs(x = NULL, y = etq, title = paste("Distribución de", etq, "por subconjunto")) +
      theme(legend.position = "none")
  })
  
  output$tabla_datos <- renderDT({
    df <- datos_cargados(); req(df)
    df_mostrar <- df; names(df_mostrar) <- etiquetas_de(df, names(df))
    datatable(df_mostrar, options = list(pageLength = 8, scrollX = TRUE))
  })
  
  # --------------------------------------------------------------------
  ## 15.7 Pestaña Resultados: Tablas 4 a 9 ----
  # Tabla 4 = métricas de prueba (fila verde = mejor modelo); Tablas 5-7 = cada
  # objetivo por subconjunto; Tabla 8 = Ho1-Hi1 a Ho3-Hi3 frente al modelo base;
  # Tabla 9 = Ho4-Hi4 (RF vs XGBoost con Diebold-Mariano).
  # --------------------------------------------------------------------
  output$tabla4 <- renderDT({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL); req(res)
    t <- do.call(rbind, lapply(res$objetivos, function(obj) {
      r <- res$resultados_por_objetivo[[obj]]
      cbind(Objetivo = res$etiquetas_objetivo[[obj]], r$metricas)
    }))
    t <- t[, c("Objetivo","Modelo","R2","MAE","RMSE","MAPE","Red_MAE_vs_base","Red_RMSE_vs_base","Cumple_criterios","EsMejor")]
    names(t) <- c("Variable objetivo","Modelo","R²","MAE","RMSE","MAPE (%)",
                  "Red. MAE (%)","Red. RMSE (%)","Cumple criterios","EsMejor")
    num <- c("R²","MAE","RMSE","MAPE (%)","Red. MAE (%)","Red. RMSE (%)")
    t[num] <- lapply(t[num], round, 4)
    dt <- datatable(t, rownames = FALSE,
                    options = list(dom = "t", pageLength = -1, scrollX = TRUE,
                                   columnDefs = list(list(visible = FALSE, targets = 9))))
    formatStyle(dt, "EsMejor", target = "row",
                backgroundColor = styleEqual(c(TRUE, FALSE), c("#E1F5EE", "white")))
  })
  
  render_tabla_subconjunto <- function(objetivo) {
    res <- tryCatch(resultado_modelos(), error = function(e) NULL); req(res)
    r <- res$resultados_por_objetivo[[objetivo]]
    if (is.null(r)) return(NULL)
    t <- r$metricas_todas
    t <- t[, c("Modelo","Subconjunto","n","R2","MAE","RMSE","MAPE")]
    names(t) <- c("Modelo","Subconjunto","n","R²","MAE","RMSE","MAPE (%)")
    t[, c("R²","MAE","RMSE","MAPE (%)")] <- lapply(t[, c("R²","MAE","RMSE","MAPE (%)")], round, 4)
    datatable(t, rownames = FALSE, options = list(dom = "t", pageLength = -1, scrollX = TRUE))
  }
  output$tabla5 <- renderDT({ render_tabla_subconjunto("recuperacion_au") })
  output$tabla6 <- renderDT({ render_tabla_subconjunto("ley_concentrado_au") })
  output$tabla7 <- renderDT({ render_tabla_subconjunto("consumo_reactivos") })
  
  output$tabla8 <- renderUI({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL); req(res)
    hg <- res$sintesis$general
    tagList(
      tabla_hip_html(tabla8_mostrar(res$sintesis$tabla8), COLS_T8, METRICAS_T8, FUSION_T8),
      tags$div(class = paste("hg", if (!hg$acepta) "no"), style = "margin-top:10px; padding:10px 14px;",
               tags$b("Hipótesis general (Ho-Hi): "), hg$decision),
      tags$div(class = "nota-hip", NOTA_TABLA8))
  })

  output$tabla9 <- renderUI({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL); req(res)
    h4 <- res$sintesis$h4
    tagList(
      tabla_hip_html(tabla9_mostrar(res$sintesis$tabla9), COLS_T9, METRICAS_T9, FUSION_T9),
      tags$div(class = paste("hg", if (!h4$acepta) "no"), style = "margin-top:10px; padding:10px 14px;",
               tags$b("Hipótesis específica 4 (Ho4-Hi4): "), h4$decision),
      tags$div(class = "nota-hip", NOTA_TABLA9))
  })

  # --------------------------------------------------------------------
  ## 15.8 Pestaña Resultados: Figuras 1 a 7 ----
  # Figura 1 = R² comparado; 2, 4, 6 = real vs. predicho; 3, 5, 7 = importancia.
  # --------------------------------------------------------------------
  output$figura1 <- renderPlot({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL); req(res)
    grafico_metricas_compacto(res)
  })
  output$figura2 <- renderPlot({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL); req(res)
    r <- res$resultados_por_objetivo[["recuperacion_au"]]; if (is.null(r)) return(NULL)
    grafico_pred_real(r, res$etiquetas_objetivo[["recuperacion_au"]], r$mejor_modelo, num_fig = 2)
  })
  output$figura3_rf <- renderPlot({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL); req(res)
    r <- res$resultados_por_objetivo[["recuperacion_au"]]; if (is.null(r)) return(NULL)
    grafico_importancia_top(r$modelos[["Random Forest"]], "RF", res$etiquetas_datos,
                            "Random Forest · Recuperación de Au", num_fig = 3)
  })
  output$figura3_xgb <- renderPlot({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL); req(res)
    r <- res$resultados_por_objetivo[["recuperacion_au"]]; if (is.null(r)) return(NULL)
    grafico_importancia_top(r$modelos[["XGBoost"]], "XGB", res$etiquetas_datos,
                            "XGBoost · Recuperación de Au", num_fig = 3)
  })
  output$figura4 <- renderPlot({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL); req(res)
    r <- res$resultados_por_objetivo[["ley_concentrado_au"]]; if (is.null(r)) return(NULL)
    grafico_pred_real(r, res$etiquetas_objetivo[["ley_concentrado_au"]], r$mejor_modelo, num_fig = 4)
  })
  output$figura5_rf <- renderPlot({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL); req(res)
    r <- res$resultados_por_objetivo[["ley_concentrado_au"]]; if (is.null(r)) return(NULL)
    grafico_importancia_top(r$modelos[["Random Forest"]], "RF", res$etiquetas_datos,
                            "Random Forest · Ley de Au en concentrado", num_fig = 5)
  })
  output$figura5_xgb <- renderPlot({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL); req(res)
    r <- res$resultados_por_objetivo[["ley_concentrado_au"]]; if (is.null(r)) return(NULL)
    grafico_importancia_top(r$modelos[["XGBoost"]], "XGB", res$etiquetas_datos,
                            "XGBoost · Ley de Au en concentrado", num_fig = 5)
  })
  output$figura6 <- renderPlot({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL); req(res)
    r <- res$resultados_por_objetivo[["consumo_reactivos"]]; if (is.null(r)) return(NULL)
    grafico_pred_real(r, res$etiquetas_objetivo[["consumo_reactivos"]], r$mejor_modelo, num_fig = 6)
  })
  output$figura7_rf <- renderPlot({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL); req(res)
    r <- res$resultados_por_objetivo[["consumo_reactivos"]]; if (is.null(r)) return(NULL)
    grafico_importancia_top(r$modelos[["Random Forest"]], "RF", res$etiquetas_datos,
                            "Random Forest · Consumo específico de reactivos", num_fig = 7)
  })
  output$figura7_xgb <- renderPlot({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL); req(res)
    r <- res$resultados_por_objetivo[["consumo_reactivos"]]; if (is.null(r)) return(NULL)
    grafico_importancia_top(r$modelos[["XGBoost"]], "XGB", res$etiquetas_datos,
                            "XGBoost · Consumo específico de reactivos", num_fig = 7)
  })
  
  # --------------------------------------------------------------------
  ## 15.9 Pestañas Discusión y Conclusiones ----
  # Texto fijo transcrito del informe: NO se recalcula si cambian los datos.
  # --------------------------------------------------------------------
  output$discusion_card <- renderUI({
    tags$div(class = "card-seccion",
             tags$h4("IV. Discusión de resultados"),
             tags$p("Cuatro condiciones permiten leer los resultados sin sobreestimarlos ni subestimarlos:"),
             tags$h5("1. El protocolo de evaluación no favoreció el resultado"),
             tags$p("La partición fue temporal: los 83 registros de prueba son los más recientes y no intervinieron en ninguna etapa del ajuste. Las particiones aleatorias en datos con dependencia temporal subestiman el error de predicción (Hewamalage et al., 2023; Lones, 2024)."),
             tags$h5("2. Los modelos siguieron un cambio de nivel de operación"),
             tags$p("El nivel medio de la recuperación se desplazó 1.06 desviaciones estándar y el del consumo 2.09 desviaciones estándar respecto a la media histórica. Los valores de consumo del periodo de prueba (0.7 a 3.0 kg/t) fueron inferiores al promedio trimestral de 3.98 kg/t, consistente con una reducción de la dosificación hacia el final del periodo (Lu et al., 2019)."),
             tags$h5("3. Random Forest vs. XGBoost: la estructura del problema decide"),
             tags$p("XGBoost ganó en consumo y ley; Random Forest en recuperación. La ventaja de XGBoost en el consumo se explica porque el boosting aproxima con eficiencia relaciones aditivas, como la suma de dosificaciones de la ecuación i3. La de Random Forest en recuperación se asocia con la reducción de varianza del promedio de árboles decorrelacionados cuando la respuesta contiene ruido de medición propio del balance metalúrgico."),
             tags$h5("4. La interpretabilidad valida la coherencia metalúrgica"),
             tags$p("En las tres variables, las dos medidas de importancia (permutación en RF, ganancia en XGBoost) coincidieron en las variables principales: el activador de la pirita, la concentración de sólidos y la composición en metales base para la recuperación; la ley de Fe, la humedad y la ley de As para la ley; y las propias dosificaciones para el consumo."),
             tags$h5("Limitaciones del estudio"),
             tags$ol(
               tags$li("Los datos provienen de una sola planta y de un periodo de nueve meses."),
               tags$li("El conjunto de prueba (n = 83) abarcó un único periodo de operación."),
               tags$li("No se documentó la disponibilidad temporal de las determinaciones del bloque ccPy."),
               tags$li("Las medidas de importancia no indican el sentido del efecto."),
               tags$li("No se estimaron intervalos de predicción que comuniquen la incertidumbre.")))
  })
  
  output$conclusiones_card <- renderUI({
    tags$div(class = "card-seccion",
             tags$h4("V. Conclusiones"),
             tags$ol(
               tags$li(tags$b("General."), " El modelo predictivo basado en IA (Random Forest y XGBoost), entrenado con 547 registros operacionales y 38 predictores, permitió predecir la flotación de relaves de lixiviación con desempeño adecuado; se aceptó la Hi y se rechazó la Ho. R² de prueba: 0.972 (recuperación), 0.989 (ley), 0.885 (consumo); MAPE de 0.96 %, 0.38 % y 9.29 %; reducciones del MAE y del RMSE superiores al 85 %."),
               tags$li(tags$b("Ho1-Hi1."), " Random Forest obtuvo R² = 0.972, MAE = 0.695 pp, RMSE = 0.850 pp y MAPE = 0.96 %, con reducciones de 87.3 % y 88.5 % frente al modelo base. XGBoost alcanzó R² = 0.965. Se aceptó la Hi1 y se rechazó la Ho1."),
               tags$li(tags$b("Ho2-Hi2."), " XGBoost obtuvo R² = 0.989, MAE = 0.029 g/t, RMSE = 0.037 g/t y MAPE = 0.38 %, con reducciones de 89.9 % y 89.4 %. Random Forest alcanzó R² = 0.985. Se aceptó la Hi2 y se rechazó la Ho2."),
               tags$li(tags$b("Ho3-Hi3."), " XGBoost obtuvo R² = 0.885, MAE = 0.146 kg/t, RMSE = 0.186 kg/t y MAPE = 9.29 %, con reducciones de 87.2 % y 85.3 %. Random Forest no cumplió los criterios (R² = 0.740; MAPE = 14.70 %). Se aceptó la Hi3 y se rechazó la Ho3 solo con XGBoost."),
               tags$li(tags$b("Ho4-Hi4."), " Ningún algoritmo fue superior en las tres variables. Random Forest superó a XGBoost en la recuperación (mayor R² y menores MAPE, MAE y RMSE; Diebold-Mariano con p = 0.007 con errores al cuadrado y p = 0.014 con errores absolutos) y XGBoost superó a Random Forest en el consumo (p < 0.001 con ambas funciones de pérdida); en la ley, Random Forest obtuvo mejores métricas y p = 0.033 con errores al cuadrado, pero p = 0.080 con errores absolutos, por lo que no hubo superioridad significativa. Como la superioridad significativa se presentó en al menos una variable, se aceptó la Hi4 y se rechazó la Ho4.")))
  })
  
  output$recomendaciones_card <- renderUI({
    tags$div(class = "card-seccion",
             tags$h4("VI. Recomendaciones"),
             tags$ol(
               tags$li(tags$b("A la gerencia de operaciones."), " Implementar los modelos como sensores blandos de apoyo a la decisión (XGBoost para ley y consumo; Random Forest o XGBoost para recuperación) durante un periodo de validación operativa en paralelo con los ensayos de laboratorio."),
               tags$li(tags$b("Al área de control de procesos."), " Establecer un esquema de reentrenamiento periódico de origen móvil (mensual o ante cambio de régimen) y un monitoreo de deriva."),
               tags$li(tags$b("Al laboratorio metalúrgico."), " Documentar la técnica analítica y el momento en que queda disponible cada determinación del bloque ccPy respecto a la ley de Au."),
               tags$li(tags$b("Al área de metalurgia."), " Complementar la importancia de variables con valores SHAP y gráficos de dependencia parcial, y validar los efectos con pruebas controladas en planta."),
               tags$li(tags$b("A futuros investigadores."), " Ampliar la base a uno o más años de operación y aplicar validación temporal de múltiples orígenes; comparar con CatBoost y modelos informados por física; estimar intervalos de predicción; incorporar caracterización mineralógica y de liberación del oro."),
               tags$li(tags$b("A la Escuela de Posgrado."), " Promover convenios con plantas concentradoras para acceder a bases de mayor extensión y frecuencia.")))
  })
  
  # --------------------------------------------------------------------
  ## 15.10 Descarga del informe PDF ----
  # Botón de descarga y generación del PDF con las pestañas Portada, Datos y Resultados.
  # --------------------------------------------------------------------
  output$botones_descarga <- renderUI({
    res <- tryCatch(resultado_modelos(), error = function(e) NULL)
    if (is.null(res)) return(NULL)
    downloadButton("descargar_pdf", "Descargar informe de comparación (PDF)",
                   class = "btn-info w-100")
  })

  # Informe PDF: reproduce el contenido de las pestañas Portada, Datos y Resultados.
  output$descargar_pdf <- downloadHandler(
    filename = function() paste0("informe_comparacion_", format(Sys.Date(), "%Y%m%d"), ".pdf"),
    content = function(file) {
      res <- resultado_modelos(); p <- res$particion
      grDevices::cairo_pdf(file, width = 11, height = 8.5, onefile = TRUE)
      on.exit(grDevices::dev.off(), add = TRUE)

      VERDE <- "#0F6E56"
      et_datos <- res$etiquetas_datos
      lab <- function(v) vapply(v, function(x)
        if (!is.null(et_datos) && x %in% names(et_datos)) unname(et_datos[[x]]) else x,
        character(1), USE.NAMES = FALSE)
      envolver <- function(x, ancho) vapply(as.character(x), function(t)
        paste(strwrap(limpiar_texto(t), width = ancho), collapse = "\n"),
        character(1), USE.NAMES = FALSE)
      fmt <- function(x, d) ifelse(is.finite(x), formatC(x, format = "f", digits = d), "—")
      texto <- function(t, tam, negrita = FALSE, color = "black")
        grid::textGrob(t, x = 0.02, hjust = 0,
                       gp = grid::gpar(fontsize = tam, col = color,
                                       fontface = if (negrita) "bold" else "plain"))
      tabla_grob <- function(df, relleno = NULL, tam = 8.5) {
        tema <- gridExtra::ttheme_default(
          base_size = tam,
          core = list(bg_params = list(fill = if (is.null(relleno)) c("white","#F9FAFB") else relleno,
                                       col = "grey85"),
                      padding = grid::unit(c(4, 3), "mm")),
          colhead = list(bg_params = list(fill = VERDE, col = "grey85"),
                         fg_params = list(col = "white", fontface = "bold"),
                         padding = grid::unit(c(4, 3), "mm")))
        gridExtra::tableGrob(df, rows = NULL, theme = tema)
      }
      alto_cm <- function(g) {
        h <- if (inherits(g, "gtable")) sum(g$heights) else grid::grobHeight(g) + grid::unit(4, "mm")
        grid::convertHeight(h, "cm", valueOnly = TRUE)
      }
      # Cada llamada empieza en una página nueva y pagina sola si no cabe.
      dibujar_bloques <- function(grupos, alto_max = 19.0) {
        paginas <- list(); actual <- list(); usado <- 0
        for (grupo in grupos) {
          if (inherits(grupo, "grob")) grupo <- list(grupo)
          h <- sum(vapply(grupo, alto_cm, numeric(1)))
          if (length(actual) > 0 && usado + h > alto_max) {
            paginas[[length(paginas) + 1]] <- actual; actual <- list(); usado <- 0
          }
          actual <- c(actual, grupo); usado <- usado + h
        }
        if (length(actual) > 0) paginas[[length(paginas) + 1]] <- actual
        for (pg in paginas) {
          alturas <- do.call(grid::unit.c,
                             c(lapply(pg, function(g) grid::unit(alto_cm(g), "cm")),
                               list(grid::unit(1, "null"))))
          grid::grid.newpage()
          grid::pushViewport(grid::viewport(width = grid::unit(10.2, "inches"),
                                            height = grid::unit(7.8, "inches")))
          grid::grid.draw(gridExtra::arrangeGrob(grobs = c(pg, list(grid::nullGrob())),
                                                 ncol = 1, heights = alturas))
          grid::popViewport()
        }
      }
      titulo_seccion <- function(t) texto(t, 16, TRUE, VERDE)
      # Orden del informe: recuperación, ley, consumo (Ho1-Hi1, Ho2-Hi2, Ho3-Hi3)
      objetivos <- c(intersect(OBJETIVOS_TESIS, res$objetivos), setdiff(res$objetivos, OBJETIVOS_TESIS))
      numeracion <- list(recuperacion_au = c(tabla = 5, pred = 2, imp = 3),
                         ley_concentrado_au = c(tabla = 6, pred = 4, imp = 5),
                         consumo_reactivos = c(tabla = 7, pred = 6, imp = 7))

      # ================================================================
      ### PDF · Portada: banner, hipótesis general, indicadores y protocolo ----
      # ================================================================
      grid::grid.newpage()
      grid::pushViewport(grid::viewport(width = grid::unit(10.2, "inches"), height = grid::unit(7.9, "inches")))
      arriba <- function(pulg) grid::unit(1, "npc") - grid::unit(pulg, "inches")
      linea <- function(t, y, tam, col = "white", negrita = FALSE, x = 0.25)
        grid::grid.text(t, x = grid::unit(x, "inches"), y = arriba(y), hjust = 0, vjust = 1,
                        gp = grid::gpar(col = col, fontsize = tam, fontface = if (negrita) "bold" else "plain"))
      colocar <- function(g, y) {
        h <- alto_cm(g)
        grid::pushViewport(grid::viewport(y = arriba(y), height = grid::unit(h, "cm"), just = "top"))
        grid::grid.draw(g); grid::popViewport()
        y + h / 2.54
      }
      grid::grid.rect(y = grid::unit(1, "npc"), height = grid::unit(1.7, "inches"), just = "top",
                      gp = grid::gpar(fill = VERDE, col = NA))
      linea("Trabajo de investigación para obtener el grado de maestro en Inteligencia Artificial", 0.22, 9)
      linea("Aplicación de machine learning para predecir la flotación de relaves de lixiviación, Lima 2026", 0.45, 15, negrita = TRUE)
      linea("Barboza Duran, Edwin Miguel · Ruiz Torres, Alfonso", 0.82, 10)
      linea("Asesor: MSc. Ruben Quispe Llacctarimay", 1.04, 10)
      linea("Universidad César Vallejo · Escuela de Posgrado · Maestría en Inteligencia Artificial · ODS 9", 1.3, 8.5)
      linea(paste("Informe de comparación generado el", format(Sys.time(), "%d/%m/%Y %H:%M")), 1.48, 8)

      # Hipótesis general Ho-Hi
      hg <- res$sintesis$general
      acepta <- hg$acepta
      grid::grid.rect(y = arriba(1.9), height = grid::unit(0.62, "inches"), just = "top",
                      gp = grid::gpar(fill = if (acepta) "#F0FDF4" else "#FEF2F2",
                                      col = if (acepta) "#BBF7D0" else "#FECACA"))
      linea("HIPÓTESIS GENERAL (Ho-Hi)", 2.0, 8.5, if (acepta) "#065F46" else "#991B1B", TRUE)
      linea(hg$decision, 2.18, 13, "#111827", TRUE)

      # Tarjetas KPI y veredicto por objetivo, en forma de tabla
      kp <- do.call(rbind, lapply(objetivos, function(obj) {
        r <- res$resultados_por_objetivo[[obj]]
        m <- r$metricas[r$metricas$Modelo == r$mejor_modelo, ]
        data.frame(Objetivo = res$etiquetas_objetivo[[obj]], Mejor = r$mejor_modelo,
                   Veredicto = if (identical(m$Cumple_criterios, "Sí")) "APTO" else "NO APTO",
                   R2 = fmt(m$R2, 3), MAPE = fmt(m$MAPE, 2),
                   RedMAE = fmt(m$Red_MAE_vs_base, 1), RedRMSE = fmt(m$Red_RMSE_vs_base, 1),
                   stringsAsFactors = FALSE)
      }))
      names(kp) <- c("Variable objetivo", "Mejor modelo", "Veredicto", "R²", "MAPE (%)",
                     "Red. MAE (%)", "Red. RMSE (%)")
      linea("Indicadores del mejor modelo en el conjunto de prueba", 2.75, 11, "#111827", TRUE)
      y <- colocar(tabla_grob(kp, relleno = ifelse(kp$Veredicto == "APTO", "#E1F5EE", "#FEE2E2"), tam = 10), 3.0)

      # Protocolo
      y <- y + 0.3
      linea("Protocolo de la tesis", y, 11, "#111827", TRUE)
      items <- c(
        sprintf("Partición: %s — entrenamiento = %s, validación = %s, prueba = %s registros.",
                p$origen, p$n_train, p$n_valid, p$n_test),
        "Modelo base: media histórica del conjunto de ajuste (entrenamiento + validación).",
        "Entrenamiento y validación: modelos ajustados solo con entrenamiento.",
        "Prueba: modelos reentrenados con entrenamiento + validación, evaluados una única vez.",
        "Criterios Ho-Hi y Ho1-Hi1 a Ho3-Hi3: R² ≥ 0,75; MAPE ≤ 10 %; MAE y RMSE menores que los del modelo base.",
        "Ho4-Hi4: mayor R² y menor MAPE, MAE y RMSE que el otro modelo, y DM con p < 0,05 con errores al cuadrado y absolutos.",
        "Las dosificaciones de los cinco reactivos se usan como predictores de las tres variables; el consumo específico total se usa como predictor de la recuperación y de la ley.")
      for (i in seq_along(items)) linea(paste0("•  ", items[i]), y + 0.08 + 0.22 * i, 9, "#374151")
      grid::popViewport()

      # ================================================================
      ### PDF · Datos: partición, descriptivos y boxplots ----
      # ================================================================
      part <- data.frame(Subconjunto = c("Entrenamiento", "Validación", "Prueba"),
                         Registros = c(p$n_train, p$n_valid, p$n_test),
                         Origen = p$origen, check.names = FALSE)
      d <- res$descriptivos
      d_obj <- d[d$Variable %in% objetivos, ]
      d_obj <- d_obj[order(match(d_obj$Variable, objetivos)), ]
      d_obj_m <- data.frame(Variable = envolver(lab(d_obj$Variable), 26), Subconjunto = d_obj$Subconjunto,
                            n = d_obj$n, `n NA` = d_obj$n_NA, Media = fmt(d_obj$Media, 3), DE = fmt(d_obj$DE, 3),
                            Min = fmt(d_obj$Min, 3), Q1 = fmt(d_obj$Q1, 3), Mediana = fmt(d_obj$Mediana, 3),
                            Q3 = fmt(d_obj$Q3, 3), Max = fmt(d_obj$Max, 3), `CV (%)` = fmt(d_obj$CV_pct, 1),
                            check.names = FALSE)
      dibujar_bloques(list(
        list(titulo_seccion("2. Datos"),
             texto("Tamaño de los subconjuntos", 11, TRUE), tabla_grob(part, tam = 9.5)),
        list(texto("Estadísticos descriptivos de las variables objetivo por subconjunto", 11, TRUE),
             tabla_grob(d_obj_m, tam = 9))))

      # Boxplots de las variables objetivo
      dp <- res$datos_particionados
      dp$Subconjunto <- factor(dp$Subconjunto, levels = c("Entrenamiento", "Validación", "Prueba"))
      cajas <- lapply(objetivos, function(v) {
        ggplot(dp, aes(x = Subconjunto, y = .data[[v]], fill = Subconjunto)) +
          geom_boxplot(alpha = 0.85, na.rm = TRUE, outlier.size = 0.8) +
          geom_jitter(width = 0.15, alpha = 0.2, size = 0.7, na.rm = TRUE) +
          scale_fill_manual(values = c("Entrenamiento" = "#2A78D6", "Validación" = "#EB6834", "Prueba" = "#1BAF7A")) +
          theme_minimal(base_size = 10) + theme(legend.position = "none") +
          labs(x = NULL, y = NULL, title = lab(v))
      })
      gridExtra::grid.arrange(grobs = cajas, ncol = length(cajas),
                              top = grid::textGrob("Distribución de las variables objetivo por subconjunto",
                                                   gp = grid::gpar(fontsize = 13, fontface = "bold", col = VERDE)))

      # Resumen de predictores (media y DE por subconjunto)
      vars_pred <- setdiff(unique(d$Variable), objetivos)
      if (length(vars_pred) > 0) {
        resumen <- do.call(rbind, lapply(vars_pred, function(v) {
          f <- function(s, col) { x <- d[d$Variable == v & d$Subconjunto == s, col]; if (length(x)) x else NA_real_ }
          data.frame(Variable = envolver(lab(v), 34),
                     `Media ent.` = fmt(f("Entrenamiento", "Media"), 3), `DE ent.` = fmt(f("Entrenamiento", "DE"), 3),
                     `Media val.` = fmt(f("Validación", "Media"), 3), `DE val.` = fmt(f("Validación", "DE"), 3),
                     `Media prueba` = fmt(f("Prueba", "Media"), 3), `DE prueba` = fmt(f("Prueba", "DE"), 3),
                     `NA total` = sum(d$n_NA[d$Variable == v]), check.names = FALSE)
        }))
        trozos <- split(seq_len(nrow(resumen)), ceiling(seq_len(nrow(resumen)) / 16))
        bloques <- lapply(seq_along(trozos), function(k) list(
          texto(sprintf("Descriptivos de las variables predictoras (%d de %d)", k, length(trozos)), 11, TRUE),
          tabla_grob(resumen[trozos[[k]], ], tam = 9.5)))
        for (b in bloques) dibujar_bloques(list(b))
      }

      # ================================================================
      ### PDF · Resultados: Tablas 4-9 y Figuras 1-7 ----
      # ================================================================
      t4 <- do.call(rbind, lapply(objetivos, function(obj) {
        cbind(Objetivo = res$etiquetas_objetivo[[obj]], res$resultados_por_objetivo[[obj]]$metricas)
      }))
      t4_mostrar <- data.frame(
        `Variable objetivo` = envolver(t4$Objetivo, 26), Modelo = t4$Modelo,
        `R²` = fmt(t4$R2, 3), MAE = fmt(t4$MAE, 3), RMSE = fmt(t4$RMSE, 3), `MAPE (%)` = fmt(t4$MAPE, 2),
        `Red. MAE (%)` = ifelse(t4$Modelo == "Modelo base", "—", fmt(t4$Red_MAE_vs_base, 1)),
        `Red. RMSE (%)` = ifelse(t4$Modelo == "Modelo base", "—", fmt(t4$Red_RMSE_vs_base, 1)),
        `Cumple` = t4$Cumple_criterios, check.names = FALSE)
      names(t4_mostrar) <- envolver(names(t4_mostrar), 12)
      dibujar_bloques(list(list(
        titulo_seccion("3. Resultados"),
        texto("Tabla 4. Desempeño predictivo de los modelos en el conjunto de prueba", 11, TRUE),
        tabla_grob(t4_mostrar, relleno = ifelse(t4$EsMejor, "#D7F0E6", "white")),
        texto("Nota. Fila verde = mejor modelo. Red. MAE y Red. RMSE: reducción porcentual respecto al modelo base. Cumple: R² ≥ 0,75; MAPE ≤ 10 %; MAE y RMSE menores que los del modelo base.", 8, color = "gray30"))))
      print(grafico_metricas_compacto(res))

      for (obj in objetivos) {
        r <- res$resultados_por_objetivo[[obj]]
        et <- res$etiquetas_objetivo[[obj]]
        nums <- numeracion[[obj]]
        tt <- r$metricas_todas
        tt_m <- data.frame(Modelo = tt$Modelo, Subconjunto = tt$Subconjunto, n = tt$n,
                           `R²` = fmt(tt$R2, 4), MAE = fmt(tt$MAE, 4), RMSE = fmt(tt$RMSE, 4),
                           `MAPE (%)` = fmt(tt$MAPE, 3), `RRMSE (%)` = fmt(tt$RRMSE, 3), check.names = FALSE)
        titulo_tabla <- if (!is.null(nums)) sprintf("Tabla %d. %s por subconjunto", nums[["tabla"]], et)
                        else paste(et, "por subconjunto")
        # Página 1 del objetivo: tabla por subconjunto + real vs. predicho
        g_tab <- tabla_grob(tt_m, tam = 9.5)
        nota_dm <- texto(paste0("Modelo seleccionado: ", r$mejor_modelo,
                                " · Diebold-Mariano (HLN) RF vs. XGBoost: e² → DM = ", fmt(r$dm$estadistico, 3),
                                ", p = ", fmt(r$dm$p_valor, 4), "; |e| → DM = ", fmt(r$dm_abs$estadistico, 3),
                                ", p = ", fmt(r$dm_abs$p_valor, 4)), 9, color = "gray30")
        g_pred <- ggplotGrob(grafico_pred_real(r, et, r$mejor_modelo,
                                               num_fig = if (!is.null(nums)) nums[["pred"]] else NULL))
        grid::grid.newpage()
        gridExtra::grid.arrange(
          grobs = list(texto(titulo_tabla, 12, TRUE, VERDE), g_tab, nota_dm, g_pred), ncol = 1,
          heights = grid::unit.c(grid::unit(1, "cm"), grid::unit(alto_cm(g_tab), "cm"),
                                 grid::unit(0.8, "cm"), grid::unit(1, "null")), newpage = FALSE)
        # Página 2 del objetivo: importancia de variables RF y XGBoost
        num_imp <- if (!is.null(nums)) nums[["imp"]] else NULL
        titulo_imp <- paste0(if (!is.null(num_imp)) paste0("Figura ", num_imp, ". ") else "",
                             "Importancia de variables: ", et)
        gridExtra::grid.arrange(
          grafico_importancia_top(r$modelos[["Random Forest"]], "RF", res$etiquetas_datos,
                                  "Random Forest (%IncMSE)"),
          grafico_importancia_top(r$modelos[["XGBoost"]], "XGB", res$etiquetas_datos,
                                  "XGBoost (Gain)"),
          ncol = 2,
          top = grid::textGrob(titulo_imp, gp = grid::gpar(fontsize = 13, fontface = "bold", col = VERDE)))
      }

      # Tablas 8 y 9: mismas tablas de la pestaña Resultados, con celdas combinadas
      hg <- res$sintesis$general
      dibujar_bloques(list(
        list(texto(TITULO_TABLA8, 11, TRUE),
             tabla_hip_grob(tabla8_mostrar(res$sintesis$tabla8), COLS_T8, METRICAS_T8, FUSION_T8,
                            anchos = c(3.9, 2.9, 2.1, 1.1, 1.8, 1.8, 1.8, 2.0, 2.6, 5.4)),
             texto(paste0("Hipótesis general (Ho-Hi): ", hg$decision, "."), 10, TRUE,
                   if (hg$acepta) "#065F46" else "#991B1B"),
             texto(NOTA_TABLA8, 8, color = "gray30")),
        list(texto(TITULO_TABLA9, 11, TRUE),
             tabla_hip_grob(tabla9_mostrar(res$sintesis$tabla9), COLS_T9, METRICAS_T9, FUSION_T9,
                            anchos = c(3.9, 2.8, 2.1, 1.1, 1.8, 1.8, 1.8, 2.0, 3.4, 5.0)),
             texto(paste0("Hipótesis específica 4 (Ho4-Hi4): ", res$sintesis$h4$decision, "."), 10, TRUE,
                   if (res$sintesis$h4$acepta) "#065F46" else "#991B1B"),
             texto(NOTA_TABLA9, 8, color = "gray30"))))
    }
  )
}

# ============================================================================
# 16. LANZAMIENTO DE LA APP ----
# Une la interfaz (sección 14) y el servidor (sección 15) y abre la aplicación.
# ============================================================================
shinyApp(ui = ui, server = server)