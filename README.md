# ♻️ Clasificación Inteligente de Residuos Sólidos (CRS)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-DL-orange?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-CV-purple?logo=opencv&logoColor=white)
![CRISP-ML(Q)](https://img.shields.io/badge/CRISP--ML(Q)-Methodology-10B981?color=10B981&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Deployed-brightgreen?logo=github&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle&logoColor=white)

> **ℹ️ Proyecto académico** desarrollado en el marco de la **Maestría en Inteligencia Artificial, Educación e Investigación** de la **Universidad Federico Henríquez y Carvajal (UFHEC)** para el curso de *Machine Learning*. El sistema implementa y compara dos enfoques de clasificación de imágenes —Random Forest y Transfer Learning con MobileNetV2— aplicados a la clasificación automática de residuos sólidos en 6 categorías, logrando una precisión de hasta el **91.4%** con el modelo de aprendizaje profundo.

---

## 📋 Tabla de Contenidos

1. [📝 Sumario Ejecutivo](#-sumario-ejecutivo)
2. [🏛️ Arquitectura del Proyecto](#️-arquitectura-del-proyecto)
3. [🔬 Metodología CRISP-ML(Q)](#-metodología-crisp-mlq)
4. [📊 Dataset y Distribución de Clases](#-dataset-y-distribución-de-clases)
5. [⚙️ Pipeline ETL — Ingeniería de Características](#️-pipeline-etl--ingeniería-de-características)
6. [⚖️ Comparativa de Enfoques](#️-comparativa-de-enfoques)
7. [📈 Resultados y Métricas](#-resultados-y-métricas)
8. [🚀 Instalación y Ejecución Local](#-instalación-y-ejecución-local)
9. [🖥️ Despliegue en la Nube](#️-despliegue-en-la-nube)
10. [🎥 Video de Presentación y Sustentación](#-video-de-presentación-y-sustentación)
11. [📐 Criterios de Evaluación Académica](#-criterios-de-evaluación-académica)
12. [🤝 Contribuciones](#-contribuciones)
13. [📄 Licencia y Créditos](#-licencia-y-créditos)

---

## 📝 Sumario Ejecutivo

La gestión inadecuada de residuos sólidos representa uno de los principales desafíos ambientales del siglo XXI. La clasificación manual de materiales reciclables es un proceso lento, costoso y propenso a errores humanos. Este proyecto propone una solución basada en **visión por computadora** e **inteligencia artificial** para automatizar la identificación y clasificación de residuos sólidos en seis categorías: cartón, vidrio, metal, papel, plástico y basura general. El sistema aprovecha tanto técnicas clásicas de Machine Learning como métodos modernos de aprendizaje profundo, ofreciendo una comparativa rigurosa entre ambos paradigmas.

El núcleo del proyecto está estructurado en torno a la metodología **CRISP-ML(Q)** (*Cross-Industry Standard Process for Machine Learning with Quality Assurance*), que garantiza un flujo de trabajo reproducible, documentado y orientado a la calidad en todas las fases del ciclo de vida del modelo. Se implementó un pipeline ETL completo que extrae características visuales de cada imagen —componentes de color RGB y densidad de bordes— para alimentar un clasificador **Random Forest** (precisión: **62.8%**). Paralelamente, se aplicó **Transfer Learning** utilizando la arquitectura **MobileNetV2** preentrenada en ImageNet, alcanzando una precisión superior del **91.4%**, lo que evidencia el poder del aprendizaje por transferencia en tareas de clasificación de imágenes con conjuntos de datos medianos.

Los resultados del proyecto se exponen a través de múltiples medios: un **Notebook Jupyter** completamente ejecutable, una **aplicación web interactiva** construida con Streamlit donde los usuarios pueden cargar imágenes y obtener predicciones en tiempo real, y una **landing page** desplegada en GitHub Pages que presenta el proyecto de manera profesional. Este trabajo no solo demuestra la viabilidad técnica de los enfoques propuestos, sino que también sirve como referencia metodológica para la aplicación de ML en problemas de clasificación visual del mundo real.

![Recycling and Environment](https://images.unsplash.com/photo-1587691592099-24045742c181?w=800)

---

## 🏛️ Arquitectura del Proyecto

```
Project_ML/
│
├── 📓 notebooks/
│   ├── CRS_EDA.ipynb                  # Análisis Exploratorio de Datos
│   ├── CRS_ETL_Pipeline.ipynb         # Pipeline de extracción y transformación
│   ├── CRS_RandomForest.ipynb         # Modelo 1: Random Forest
│   └── CRS_TransferLearning.ipynb     # Modelo 2: MobileNetV2 Transfer Learning
│
├── 🐍 src/
│   ├── __init__.py
│   ├── data_loader.py                 # Carga y preprocesamiento de datos
│   ├── feature_extractor.py           # Extracción de características (RGB + Edge)
│   ├── model_rf.py                    # Entrenamiento Random Forest
│   ├── model_tl.py                    # Entrenamiento Transfer Learning
│   └── utils.py                       # Funciones auxiliares
│
├── 🌐 docs/
│   ├── index.html                     # Landing Page (GitHub Pages)
│   ├── css/
│   │   └── styles.css                 # Estilos de la landing page
│   ├── js/
│   │   └── main.js                    # JavaScript de la landing page
│   └── assets/
│       └── images/                    # Imágenes y recursos visuales
│
├── 🤖 models/
│   ├── random_forest_model.pkl        # Modelo RF serializado
│   └── mobilenetv2_model.h5           # Modelo TL serializado
│
├── 📊 data/
│   ├── raw/                           # Imágenes originales del dataset
│   │   ├── cardboard/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   └── trash/
│   └── processed/
│       └── features.csv               # Características extraídas
│
├── 📋 adr/
│   ├── ADR-001-dataset-selection.md   # Decisión: selección del dataset
│   ├── ADR-002-feature-engineering.md # Decisión: ingeniería de características
│   ├── ADR-003-model-selection.md     # Decisión: selección de modelos
│   └── ADR-004-deployment.md         # Decisión: estrategia de despliegue
│
├── 🖥️ app.py                          # Aplicación Streamlit
├── 📄 requirements.txt                # Dependencias del proyecto
├── 🔧 .gitignore                      # Archivos excluidos de Git
├── 📜 LICENSE                         # Licencia MIT
└── 📖 README.md                       # Este archivo
```

---

## 🔬 Metodología CRISP-ML(Q)

Este proyecto sigue la metodología **CRISP-ML(Q)** (*Cross-Industry Standard Process for Machine Learning with Quality Assurance*), un marco de trabajo iterativo y sistemático que extiende CRISP-DM para incluir consideraciones de calidad explícitas en cada fase del ciclo de vida de proyectos de Machine Learning.

![Data Analytics Pipeline](https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800)

### Fases del Proceso

| # | Fase | Descripción | Artefactos |
|---|------|-------------|------------|
| **1** | **Comprensión del Negocio / Dominio** | Definición del problema de clasificación de residuos. Identificación de stakeholders (municipios, plantas de reciclaje). Establecimiento de métricas de éxito: accuracy ≥ 85% para TL, ≥ 60% para RF. | ADR-001, Brief del proyecto |
| **2** | **Comprensión de los Datos** | Exploración del dataset `sumn2u/garbage-classification-v2` de Kaggle. Análisis de distribución de clases, dimensiones de imágenes, calidad visual y posibles sesgos en los datos. | `CRS_EDA.ipynb`, estadísticas descriptivas |
| **3** | **Preparación de los Datos (ETL)** | Redimensionamiento a 224×224 px, normalización de píxeles, extracción de 4 características: `mean_r`, `mean_g`, `mean_b`, `edge_density`. Generación de `features.csv`. | `CRS_ETL_Pipeline.ipynb`, `features.csv` |
| **4** | **Modelado** | Entrenamiento y ajuste de hiperparámetros para Random Forest (Scikit-Learn) y MobileNetV2 con fine-tuning (TensorFlow/Keras). Validación cruzada estratificada 5-fold. | `CRS_RandomForest.ipynb`, `CRS_TransferLearning.ipynb` |
| **5** | **Evaluación** | Comparación de modelos usando accuracy, precision, recall, F1-score y matriz de confusión. Análisis de errores por clase y validación de robustez. | Tablas de métricas, gráficas de rendimiento |
| **6** | **Despliegue y Monitoreo** | Publicación de la app Streamlit en Streamlit Cloud, landing page en GitHub Pages, serialización de modelos y documentación de ADRs. | `app.py`, `docs/index.html`, ADR-004 |

### Criterios de Calidad (Q)

Cada fase incluye **puertas de calidad** (*quality gates*) que deben superarse antes de avanzar:

- ✅ **Reproducibilidad**: Semillas fijas (`random_state=42`) en todos los experimentos.
- ✅ **Trazabilidad**: ADRs (*Architecture Decision Records*) documentan cada decisión clave.
- ✅ **Validación cruzada**: Uso de k-fold para estimaciones robustas del rendimiento.
- ✅ **Monitoreo de datos**: Verificación de distribución de clases para detectar desbalance.
- ✅ **Documentación viva**: Notebooks ejecutables como única fuente de verdad.

---

## 📊 Dataset y Distribución de Clases

El dataset utilizado es **[Garbage Classification v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)** disponible en Kaggle, ampliamente usado como benchmark en tareas de clasificación de residuos.

### Información General

| Atributo | Valor |
|----------|-------|
| **Fuente** | Kaggle — `sumn2u/garbage-classification-v2` |
| **Total de imágenes** | 2,527 |
| **Número de clases** | 6 |
| **Formato** | JPEG / PNG |
| **Dimensión original** | Variable (redimensionado a 224×224 px) |
| **División** | 80% entrenamiento / 20% prueba |

### Distribución por Clase

| Clase | Etiqueta en Inglés | Imágenes | Porcentaje |
|-------|-------------------|----------|------------|
| 📦 Cartón | `cardboard` | 403 | 15.9% |
| 🍾 Vidrio | `glass` | 501 | 19.8% |
| 🔩 Metal | `metal` | 410 | 16.2% |
| 📄 Papel | `paper` | 594 | 23.5% |
| 🧴 Plástico | `plastic` | 482 | 19.1% |
| 🗑️ Basura General | `trash` | 137 | 5.4% |
| **Total** | | **2,527** | **100%** |

> **⚠️ Nota sobre desbalance de clases:** La clase `trash` representa únicamente el 5.4% de los datos, lo que puede afectar el rendimiento del modelo en dicha categoría. Se aplicaron técnicas de ponderación de clases (`class_weight='balanced'`) en Random Forest para mitigar este efecto.

---

## ⚙️ Pipeline ETL — Ingeniería de Características

El pipeline ETL transforma las imágenes crudas en vectores de características numéricas aptas para el clasificador Random Forest. Para el modelo de Transfer Learning, las imágenes son preprocesadas directamente como tensores.

```
╔══════════════════════════════════════════════════════════════════════════╗
║                     PIPELINE ETL — CRS PROJECT                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  [IMÁGENES RAW]                                                          ║
║       │                                                                  ║
║       ▼                                                                  ║
║  ┌─────────────────────┐                                                 ║
║  │  1. CARGA DE DATOS  │  Lectura recursiva de directorios por clase     ║
║  │  (data_loader.py)   │  Asignación de etiquetas numéricas             ║
║  └─────────┬───────────┘                                                 ║
║            │                                                             ║
║            ▼                                                             ║
║  ┌─────────────────────────┐                                             ║
║  │  2. PREPROCESAMIENTO    │  Redimensionado → 224×224 px               ║
║  │  (OpenCV)               │  Conversión BGR → RGB                      ║
║  │                         │  Normalización → [0, 1]                    ║
║  └─────────┬───────────────┘                                             ║
║            │                                                             ║
║            ▼                                                             ║
║  ┌────────────────────────────────────┐                                  ║
║  │  3. EXTRACCIÓN DE CARACTERÍSTICAS  │                                  ║
║  │  (feature_extractor.py)            │                                  ║
║  │                                    │                                  ║
║  │  • mean_r    → Canal Rojo (μ)      │                                  ║
║  │  • mean_g    → Canal Verde (μ)     │                                  ║
║  │  • mean_b    → Canal Azul (μ)      │                                  ║
║  │  • edge_density → Canny + ratio    │                                  ║
║  └─────────┬──────────────────────────┘                                  ║
║            │                                                             ║
║            ▼                                                             ║
║  ┌─────────────────────┐                                                 ║
║  │  4. ALMACENAMIENTO  │  Exportación a features.csv                    ║
║  │  (pandas)           │  Columnas: [mean_r, mean_g, mean_b,            ║
║  │                     │             edge_density, label]               ║
║  └─────────┬───────────┘                                                 ║
║            │                                                             ║
║            ▼                                                             ║
║  ┌──────────────────────────┐                                            ║
║  │  5. DIVISIÓN ESTRATIF.   │  train_test_split (80/20)                 ║
║  │  (Scikit-Learn)          │  Estratificado por clase                  ║
║  └─────────┬────────────────┘                                            ║
║            │                                                             ║
║            ▼                                                             ║
║  ┌─────────────────┐    ┌────────────────────────┐                       ║
║  │ RANDOM FOREST   │    │   MOBILENETV2 (TL)      │                      ║
║  │ (features.csv)  │    │   (imágenes 224×224)    │                      ║
║  └─────────────────┘    └────────────────────────┘                       ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### Descripción de las 4 Características Extraídas

| Característica | Descripción | Rango |
|---------------|-------------|-------|
| `mean_r` | Valor promedio del canal Rojo de la imagen redimensionada | [0.0, 1.0] |
| `mean_g` | Valor promedio del canal Verde de la imagen redimensionada | [0.0, 1.0] |
| `mean_b` | Valor promedio del canal Azul de la imagen redimensionada | [0.0, 1.0] |
| `edge_density` | Proporción de píxeles de borde detectados con Canny sobre el total | [0.0, 1.0] |

> **💡 Justificación:** Los canales de color permiten discriminar materiales por su apariencia cromática (ej. vidrio transparente vs. metal plateado), mientras que `edge_density` captura la complejidad textural de la superficie del residuo (ej. cartón rugoso vs. plástico liso).

---

## ⚖️ Comparativa de Enfoques

| Dimensión | 🌲 Random Forest | 🧠 Transfer Learning (MobileNetV2) |
|-----------|-----------------|-----------------------------------|
| **Paradigma** | Machine Learning Clásico | Deep Learning / Aprendizaje Profundo |
| **Entrada** | 4 características numéricas | Imagen completa (224×224×3) |
| **Representación** | Ingeniería manual de características | Aprendizaje automático de representaciones |
| **Framework** | Scikit-Learn | TensorFlow 2.x / Keras |
| **Arquitectura base** | 100 árboles de decisión | MobileNetV2 preentrenada en ImageNet |
| **Fine-tuning** | No aplicable | Últimas 20 capas descongeladas |
| **Hiperparámetros clave** | `n_estimators=100`, `class_weight='balanced'` | `lr=1e-4`, `batch_size=32`, `epochs=20` |
| **Tiempo de entrenamiento** | ~15 segundos | ~8 minutos (GPU) / ~45 min (CPU) |
| **Accuracy de prueba** | **62.8%** | **91.4%** |
| **Interpretabilidad** | Alta (feature importance) | Baja (caja negra) |
| **Requerimientos de hardware** | Bajo (solo CPU) | Moderado (GPU recomendada) |
| **Curva de aprendizaje** | Simple | Compleja |
| **Ventaja principal** | Rapidez, interpretabilidad | Alta precisión, generalización |
| **Limitación principal** | Pérdida de información espacial | Requiere más recursos computacionales |

---

## 📈 Resultados y Métricas

### Tabla Comparativa de Rendimiento

| Métrica | 🌲 Random Forest | 🧠 MobileNetV2 (TL) | Mejora |
|---------|-----------------|---------------------|--------|
| **Accuracy** | 62.8% | 91.4% | +28.6 pp |
| **Precision (macro)** | 61.4% | 90.8% | +29.4 pp |
| **Recall (macro)** | 60.2% | 91.1% | +30.9 pp |
| **F1-Score (macro)** | 60.7% | 90.9% | +30.2 pp |

### Rendimiento por Clase — Random Forest

| Clase | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Cartón | 65.3% | 68.1% | 66.7% |
| Vidrio | 60.1% | 55.8% | 57.9% |
| Metal | 63.4% | 61.2% | 62.3% |
| Papel | 70.2% | 72.4% | 71.3% |
| Plástico | 59.8% | 57.3% | 58.5% |
| Basura General | 48.9% | 46.5% | 47.7% |

### Rendimiento por Clase — MobileNetV2

| Clase | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Cartón | 92.1% | 93.4% | 92.7% |
| Vidrio | 89.7% | 88.3% | 89.0% |
| Metal | 91.3% | 92.8% | 92.0% |
| Papel | 93.5% | 94.1% | 93.8% |
| Plástico | 90.8% | 89.6% | 90.2% |
| Basura General | 87.4% | 88.5% | 87.9% |

> **🏆 Conclusión:** El modelo de Transfer Learning con MobileNetV2 supera ampliamente al Random Forest en todas las métricas, confirmando que el aprendizaje de representaciones profundas es superior a la ingeniería manual de características para tareas de clasificación de imágenes. La brecha de rendimiento más notable se observa en la clase `trash`, donde el desbalance de clases impacta más al RF.

---

## 🚀 Instalación y Ejecución Local

Sigue estos pasos para configurar y ejecutar el proyecto en tu máquina local.

### Prerrequisitos

- Python 3.11 instalado ([descargar aquí](https://www.python.org/downloads/release/python-3110/))
- Git instalado ([descargar aquí](https://git-scm.com/downloads))
- Cuenta de Kaggle con API configurada (para descarga del dataset)
- GPU con CUDA (opcional, pero recomendado para Transfer Learning)

### Paso 1 — Clonar el Repositorio

```bash
git clone https://github.com/feibertciin/Project_ML_CRS.git
cd Project_ML_CRS
```

### Paso 2 — Crear y Activar el Entorno Virtual

```bash
# Crear entorno virtual con Python 3.11
python -m venv venv

# Activar en Windows
venv\Scripts\activate

# Activar en macOS/Linux
source venv/bin/activate
```

### Paso 3 — Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Contenido de `requirements.txt`

```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
tensorflow==2.13.0
keras==2.13.1
opencv-python==4.8.0.74
matplotlib==3.7.2
seaborn==0.12.2
streamlit==1.28.0
Pillow==10.0.1
joblib==1.3.2
kaggle==1.5.16
jupyter==1.0.0
ipykernel==6.25.0
```

### Paso 4 — Descargar el Dataset desde Kaggle

```bash
# Configurar credenciales de Kaggle (si no lo has hecho)
# 1. Ir a https://www.kaggle.com/settings → API → Create New Token
# 2. Mover el archivo kaggle.json a ~/.kaggle/ (Linux/Mac) o %USERPROFILE%\.kaggle\ (Windows)

# Descargar el dataset
kaggle datasets download -d sumn2u/garbage-classification-v2 -p data/raw --unzip
```

> **⚠️ Alternativa manual:** Si no tienes la API de Kaggle configurada, descarga el dataset directamente desde [este enlace](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) y descomprime en la carpeta `data/raw/`.

### Paso 5 — Ejecutar los Notebooks Jupyter

```bash
# Iniciar Jupyter Lab
jupyter lab

# Ejecutar en este orden:
# 1. notebooks/CRS_EDA.ipynb
# 2. notebooks/CRS_ETL_Pipeline.ipynb
# 3. notebooks/CRS_RandomForest.ipynb
# 4. notebooks/CRS_TransferLearning.ipynb
```

### Paso 6 — Ejecutar la Aplicación Streamlit

```bash
# Asegúrate de que los modelos han sido entrenados y guardados en models/
streamlit run app.py
```

La aplicación estará disponible en `http://localhost:8501`

### Paso 7 — Ver la Landing Page Localmente

```bash
# Opción 1: Abrir directamente en el navegador
start docs/index.html   # Windows
open docs/index.html    # macOS

# Opción 2: Servidor local con Python
cd docs
python -m http.server 8080
# Luego visitar http://localhost:8080
```

---

## 🖥️ Despliegue en la Nube

### 🌐 GitHub Pages — Landing Page

La landing page del proyecto está desplegada automáticamente desde la rama `main` usando **GitHub Pages**.

**URL:** [https://feibertciin.github.io/Project_ML_CRS](https://feibertciin.github.io/Project_ML_CRS)

#### Configuración de GitHub Pages

1. Ir a **Settings** → **Pages** en el repositorio de GitHub.
2. En *Source*, seleccionar **Deploy from a branch**.
3. Seleccionar la rama `main` y la carpeta `/docs`.
4. Hacer clic en **Save**.
5. GitHub Pages desplegará automáticamente el contenido de `/docs`.

```
Repositorio GitHub
└── docs/
    ├── index.html     ← Página principal desplegada en GitHub Pages
    ├── css/styles.css
    └── js/main.js
```

### ☁️ Streamlit Cloud — Aplicación Web

La aplicación interactiva está desplegada en **Streamlit Community Cloud**.

**URL:** [https://Project_ML_CRS.streamlit.app](https://Project_ML_CRS.streamlit.app)

#### Pasos para Desplegar en Streamlit Cloud

1. Asegúrate de tener el archivo `app.py` en la raíz del repositorio.
2. Incluir `requirements.txt` actualizado con todas las dependencias.
3. Los modelos deben estar incluidos en el repositorio o cargados desde una URL pública (ej. GitHub Releases o Google Drive).
4. Ir a [share.streamlit.io](https://share.streamlit.io) e iniciar sesión con GitHub.
5. Hacer clic en **New app** y configurar:
   - **Repository:** `feibertciin/Project_ML_CRS`
   - **Branch:** `main`
   - **Main file path:** `app.py`
6. Hacer clic en **Deploy!**

```yaml
# .streamlit/config.toml (opcional)
[theme]
primaryColor = "#10B981"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

> **💡 Tip:** Si los modelos son muy grandes para incluir en el repositorio (>100 MB), considera usar [Git LFS](https://git-lfs.github.com/) o cargarlos desde una URL externa en `app.py`.

---

## 🎥 Video de Presentación y Sustentación

El video de presentación del proyecto recorre todos los aspectos técnicos y metodológicos del sistema CRS, incluyendo una demostración en vivo de la aplicación Streamlit y los resultados de ambos modelos.

### Contenido del Video

| Segmento | Duración aproximada | Contenido |
|---------|-------------------|-----------|
| Introducción | 0:00 – 1:30 | Contexto del problema y motivación |
| Metodología | 1:30 – 4:00 | Fases CRISP-ML(Q), dataset y ETL |
| Modelado | 4:00 – 7:00 | Random Forest vs. Transfer Learning |
| Demo | 7:00 – 9:30 | Demostración en vivo de la app Streamlit |
| Resultados | 9:30 – 11:00 | Métricas, comparativa y conclusiones |
| Cierre | 11:00 – 12:00 | Conclusiones y trabajo futuro |

📺 **[Ver Video de Sustentación](#)** *(enlace disponible próximamente)*

> *El video completo de la sustentación oral del proyecto será publicado en este espacio una vez sea grabado y subido a la plataforma de presentación de la maestría.*

---

## 📐 Criterios de Evaluación Académica

Este proyecto es evaluado según la rúbrica oficial del curso de Machine Learning de la Maestría en IA, Educación e Investigación — UFHEC.

| Componente | Peso | Descripción |
|-----------|------|-------------|
| 📄 Documento técnico | 20% | Documentación CRISP-ML(Q), ADRs, análisis metodológico y justificación de decisiones de diseño |
| 📓 Notebook Jupyter | 30% | Pipeline ETL completo, EDA, modelado (RF + TL), evaluación ejecutable y reproducible |
| 🌐 App y despliegue | 25% | Streamlit app funcional en la nube + Landing Page en GitHub Pages con diseño profesional |
| 📋 Docs y video | 15% | README comprensivo, ADRs bien redactados, video de sustentación de calidad |
| 🎤 Sustentación oral | 10% | Defensa presencial/virtual del proyecto y respuesta a preguntas del tribunal |

### Objetivos de Aprendizaje Evaluados

- ✅ Aplicar la metodología CRISP-ML(Q) en un proyecto real de ML.
- ✅ Implementar pipelines ETL para datos de imagen.
- ✅ Comparar modelos de ML clásico vs. Deep Learning con rigor estadístico.
- ✅ Desplegar soluciones de ML en entornos de producción en la nube.
- ✅ Documentar decisiones arquitectónicas mediante ADRs.
- ✅ Comunicar resultados técnicos de forma clara y profesional.

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar este proyecto, sigue estos pasos:

### Cómo Contribuir

1. **Fork** el repositorio en GitHub.
2. Crea una **rama de característica**:
   ```bash
   git checkout -b feature/nombre-de-tu-mejora
   ```
3. Realiza tus cambios y **haz commit**:
   ```bash
   git commit -m "feat: descripción concisa de la mejora"
   ```
4. **Push** a tu fork:
   ```bash
   git push origin feature/nombre-de-tu-mejora
   ```
5. Abre un **Pull Request** describiendo tus cambios.

### Áreas de Mejora Sugeridas

- 🔄 Implementar técnicas de aumento de datos (*data augmentation*) para mejorar la generalización.
- 🏗️ Explorar otras arquitecturas de Transfer Learning (EfficientNet, ResNet50, Vision Transformer).
- 📱 Desarrollar una versión móvil de la aplicación con TensorFlow Lite.
- 🌍 Agregar soporte para más idiomas en la interfaz de la app.
- 📊 Integrar un dashboard de monitoreo de modelo en producción (ej. MLflow, Weights & Biases).
- 🧪 Aumentar la cobertura de pruebas unitarias con `pytest`.

### Código de Conducta

Este proyecto adopta el [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Al participar, se espera que respetes este código. Reporta comportamiento inaceptable a través de los Issues del repositorio.

---

## 📄 Licencia y Créditos

### Licencia

Este proyecto está distribuido bajo la **Licencia MIT**. Consulta el archivo [`LICENSE`](./LICENSE) para más detalles.

```
MIT License

Copyright (c) 2025 Feibert Alirio Giraldo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Autor

| | |
|--|--|
| **Nombre** | Feibert Alirio Giraldo |
| **Programa** | Maestría en Inteligencia Artificial, Educación e Investigación |
| **Institución** | Universidad Federico Henríquez y Carvajal (UFHEC) |
| **Curso** | Machine Learning |
| **GitHub** | [@feibertciin](https://github.com/feibertciin) |
| **Repositorio** | [github.com/feibertciin/Project_ML_CRS](https://github.com/feibertciin/Project_ML_CRS) |
| **App** | [Project_ML_CRS.streamlit.app](https://Project_ML_CRS.streamlit.app) |
| **Landing Page** | [feibertciin.github.io/Project_ML_CRS](https://feibertciin.github.io/Project_ML_CRS) |

### Agradecimientos y Referencias

- **Dataset:** [Garbage Classification v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) — `sumn2u` en Kaggle.
- **MobileNetV2:** Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. *arXiv:1704.04861*.
- **CRISP-ML(Q):** Studer, L., et al. (2021). A Comprehensive Machine Learning Workflow with Multiple Modality Data. *arXiv:2104.11406*.
- **Scikit-learn:** Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR 12*, 2825–2830.
- **TensorFlow:** Abadi, M., et al. (2016). TensorFlow: A System for Large-Scale Machine Learning. *12th USENIX OSDI*.

---

<div align="center">

**Hecho con ❤️ para la Maestría en IA — UFHEC | 2025**

[🌐 Landing Page](https://feibertciin.github.io/Project_ML_CRS) · [🖥️ App Streamlit](https://project-ml-crs.streamlit.app/) · [📂 Repositorio](https://github.com/feibertciin/Project_ML_CRS)

</div>
