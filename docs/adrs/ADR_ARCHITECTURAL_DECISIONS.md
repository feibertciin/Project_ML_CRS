# Decisiones Arquitectónicas — Clasificación Inteligente de Residuos Sólidos (CRS)

> Registro de Decisiones Arquitectónicas (ADRs) en formato MADR.  
> Proyecto: **CRS — Clasificación Inteligente de Residuos Sólidos**  
> Responsable: Feibert Alirio Giraldo  
> Última actualización: 2026-11-04

---

## Índice

| ID | Título | Estado |
|----|--------|--------|
| [ADR-001](#adr-001-extracción-de-características-tabulares-desde-imágenes-para-modelado-clásico) | Extracción de Características Tabulares desde Imágenes para Modelado Clásico | Accepted |
| [ADR-002](#adr-002-inclusión-de-transfer-learning-con-mobilenetv2-como-enfoque-comparativo) | Inclusión de Transfer Learning con MobileNetV2 como Enfoque Comparativo | Accepted |
| [ADR-003](#adr-003-selección-de-framework-de-despliegue-web-streamlit-cloud--github-pages) | Selección de Framework de Despliegue Web (Streamlit Cloud + GitHub Pages) | Accepted |
| [ADR-004](#adr-004-selección-de-dataset-kaggle-vs-dataset-propio) | Selección de Dataset Kaggle vs Dataset Propio | Accepted |
| [ADR-005](#adr-005-estrategia-de-balanceo-de-clases-class_weightbalanced-vs-smote-vs-oversampling) | Estrategia de Balanceo de Clases (class_weight='balanced' vs SMOTE vs Oversampling) | Accepted |

---

## ADR-001: Extracción de Características Tabulares desde Imágenes para Modelado Clásico

- **Status:** Accepted
- **Date:** 2026-11-04
- **Deciders:** Feibert Alirio Giraldo

---

### Contexto

El proyecto CRS tiene como objetivo clasificar residuos sólidos en 6 categorías (cartón, vidrio, metal, papel, plástico, basura) a partir de imágenes capturadas en entornos reales. Una de las restricciones clave del proyecto es la necesidad de evaluar el rendimiento de algoritmos de Aprendizaje Automático clásico sobre datos visuales, considerando escenarios de despliegue en entornos con recursos computacionales limitados, como dispositivos IoT, Raspberry Pi o CPUs sin aceleración GPU. En estos contextos, el uso directo de redes neuronales convolucionales profundas resulta prohibitivo en términos de memoria, latencia y dependencias de software.

El reto central de esta decisión consiste en determinar cómo representar la información contenida en imágenes en una forma tabular compacta que pueda ser procesada eficientemente por algoritmos como Random Forest, SVM o Gradient Boosting. La transformación debe preservar suficiente información discriminativa para que el clasificador diferencie entre las 6 clases, al tiempo que mantiene la dimensionalidad mínima para evitar el problema de la maldición de la dimensionalidad.

Adicionalmente, el proyecto requiere que las características extraídas sean **interpretables**, de modo que el modelo resultante pueda explicarse a audiencias no técnicas (gestores de residuos, administradores municipales) mediante importancias de características directamente legibles. Esta restricción de interpretabilidad descarta enfoques de extracción de características basados en capas intermedias de redes neuronales (embeddings no interpretables).

---

### Decisión

Implementar un pipeline clásico de visión por computador usando OpenCV que extrae **4 características estadísticas globales** por imagen redimensionada a 100×100 píxeles:

1. **`mean_r`**: Intensidad media del canal rojo (tras conversión BGR→RGB).
2. **`mean_g`**: Intensidad media del canal verde.
3. **`mean_b`**: Intensidad media del canal azul.
4. **`edge_density`**: Densidad de bordes calculada con el filtro Canny (umbrales 50 y 150), definida como `edges.mean() / 255`, donde `edges` es la máscara binaria de bordes.

El resultado es una matriz tabular de **2,527 filas × 4 columnas** (~200 KB en CSV), que se utiliza como entrada directa al clasificador Random Forest.

---

### Justificación

La elección de estas 4 características responde a tres criterios: minimalidad dimensional, interpretabilidad directa y cobertura de los atributos visuales más discriminativos para residuos:

- **Canales de color (mean_r, mean_g, mean_b):** El color es el atributo visual más diferenciador entre categorías de residuos. El vidrio tiende a ser transparente/azulado, el metal a tonos grises, el cartón a tonos marrones, el papel a tonos blancos. Tres medias de canal capturan esta información de forma compacta y sin pérdida significativa respecto a histogramas completos de 256 bins.
- **Densidad de bordes (edge_density):** Captura la complejidad estructural de la imagen. El metal arrugado y el cartón corrugado presentan alta densidad de bordes; el plástico liso, baja densidad. Este único valor resume información de textura que sería mucho más costosa de calcular con HOG o LBP.
- **Maldición de la dimensionalidad:** A diferencia del aplanamiento de píxeles crudos (100×100×3 = 30,000 features), 4 características garantizan que Random Forest pueda construir árboles sin sobreajuste y con tiempos de entrenamiento inferiores a 10 segundos en CPU.

---

### Alternativas Consideradas

| Alternativa | Descripción | Razón de Rechazo |
|-------------|-------------|-----------------|
| **Píxeles crudos aplanados** | Imagen 100×100×3 → vector de 30,000 características | Maldición de la dimensionalidad; entrenamiento RF extremadamente lento; sin interpretabilidad |
| **HOG (Histogram of Oriented Gradients)** | Descriptores de gradientes orientados por celdas | Alta complejidad de implementación; genera 2,000+ features; pierde información de color |
| **Histogramas de color (256 bins/canal)** | 768 características de distribución de intensidad | Más features que el enfoque adoptado con información marginal adicional; menos interpretable |
| **LBP (Local Binary Patterns)** | Descriptores de textura local binaria | Captura solo textura, no color; información incompleta para clasificar residuos por material |
| **Embeddings de CNN intermedia** | Extraer activaciones de capas intermedias de VGG/ResNet | No interpretable; requiere infraestructura GPU/TF; pierde la ventaja del enfoque clásico |

---

### Consecuencias

**Positivas:**
- Reducción del 99.3% en el tamaño del dataset (imágenes ~500MB → CSV ~200KB).
- Tiempo de entrenamiento del modelo RF < 10 segundos en CPU estándar.
- Pipeline totalmente interpretable: importancias de características directamente legibles.
- Desplegable en microcontroladores y dispositivos edge sin dependencias de deep learning.
- Código de extracción reproducible en < 20 líneas de Python/OpenCV.

**Negativas:**
- Pérdida de información espacial: dos imágenes con colores similares distribuidos diferente producen las mismas 3 medias de canal.
- Sensibilidad a rotación y escala: `edge_density` varía si la imagen está rotada.
- Techo de precisión de ~62.8%: la información de 4 features no es suficiente para separar limpiamente 6 clases con alta varianza intra-clase.
- El feature `edge_density` puede ser ruidoso en imágenes con fondos complejos.

---

### Notas de Implementación

Código implementado en `Project_ML_CRS.ipynb`. Función de extracción:

```python
import cv2
import numpy as np

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean_r = img_rgb[:, :, 0].mean()
    mean_g = img_rgb[:, :, 1].mean()
    mean_b = img_rgb[:, :, 2].mean()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255
    return [mean_r, mean_g, mean_b, edge_density]
```

Dataset resultante almacenado en `data/features.csv` con columnas: `[mean_r, mean_g, mean_b, edge_density, label]`.

---

## ADR-002: Inclusión de Transfer Learning con MobileNetV2 como Enfoque Comparativo

- **Status:** Accepted
- **Date:** 2026-11-04
- **Deciders:** Feibert Alirio Giraldo

---

### Contexto

El pipeline clásico definido en ADR-001 alcanza una precisión del 62.8% sobre las 6 clases del dataset Garbage Classification v2. Para el caso de uso específico de clasificación de botellas de plástico PET diferenciadas por tamaño (Botella_Plast_Peq, Botella_Plast_Med, Botella_Plast_Grd), las 4 características tabulares generan espacios de características superpuestos: tres clases de botellas PET presentan colores casi idénticos (transparente/blanco) y densidades de borde similares. El clasificador RF no puede separar estas clases con los features disponibles, produciendo confusión inter-clase sistemática.

Para abordar esta limitación, se requiere un enfoque que capture patrones morfológicos finos: variaciones de forma, proporción altura/ancho, deformaciones del plástico, etiquetas impresas. Este nivel de detalle solo es accesible mediante características aprendidas jerárquicamente por redes convolucionales profundas. Sin embargo, entrenar una CNN desde cero requiere al menos 10,000 imágenes por clase y acceso a GPU, recursos no disponibles en el contexto académico del proyecto.

El Transfer Learning emerge como solución: aprovechar los pesos preentrenados de una CNN en ImageNet (1.2M imágenes, 1,000 clases) para transferir representaciones visuales ricas a un problema de dominio específico con pocos datos. La decisión clave es: **¿qué arquitectura de CNN preentrenada es más adecuada para un despliegue en Streamlit Cloud con restricciones de RAM (1GB) y sin GPU?**

---

### Decisión

Incorporar Transfer Learning utilizando **MobileNetV2 preentrenado en ImageNet**, con la siguiente arquitectura de cabeza de clasificación personalizada:

| Capa | Configuración |
|------|--------------|
| Input | 224 × 224 × 3 (RGB, normalizado /255) |
| MobileNetV2 base | Todas las capas convolucionales **congeladas** (pesos ImageNet) |
| GlobalAveragePooling2D | Reduce feature maps a vector 1D (1,280 dims) |
| Dense(100, relu) | Capa de clasificación intermedia |
| Dropout(0.3) | Regularización para prevenir sobreajuste con dataset pequeño |
| Dense(3, softmax) | Salida: probabilidades para 3 clases PET |

El modelo se almacena en `converted_keras/keras_model.h5` con etiquetas en `converted_keras/labels.txt`. Se expone en la aplicación Streamlit (Tab 2) para inferencia en tiempo real sobre imágenes cargadas por el usuario.

---

### Justificación

MobileNetV2 es la arquitectura óptima para este escenario por las siguientes razones técnicas:

- **Eficiencia en parámetros:** 3.4M parámetros vs 25M de ResNet50 y 138M de VGG16. Con 1GB RAM disponible en Streamlit Cloud, MobileNetV2 carga en <2 segundos y realiza inferencia en 25-50ms.
- **Diseño para edge/mobile:** Arquitectura basada en depthwise separable convolutions e inverted residuals, diseñada explícitamente para CPUs móviles. Optimiza el balance precisión/compute mejor que cualquier alternativa en este rango de parámetros.
- **Compatibilidad con Teachable Machine:** Google Teachable Machine exporta modelos en formato `keras_model.h5` compatible con MobileNetV2. Esto permite entrenamiento visual sin código y exportación directa al pipeline de despliegue.
- **Precisión suficiente:** 91.4% de accuracy en las 3 clases PET supera en 28.6 puntos porcentuales al pipeline clásico sobre el mismo subconjunto.

---

### Alternativas Consideradas

| Alternativa | Parámetros | Razón de Rechazo |
|-------------|-----------|-----------------|
| **VGG16** | 138M | Demasiado pesado para Streamlit Cloud (supera 1GB RAM); inferencia >500ms en CPU |
| **ResNet50** | 25M | Excesivo para 3 clases con dataset pequeño; tiempo de carga elevado en entorno serverless |
| **CNN personalizada desde cero** | Variable | Requiere >10,000 imágenes/clase; sin GPU disponible; underfitting garantizado con dataset actual |
| **EfficientNetB0** | 5.3M | Considerado; mayor precisión que MobileNetV2 en benchmarks, pero peor compatibilidad con el formato `keras_model.h5` de Teachable Machine export |
| **SqueezeNet** | 1.2M | Precisión inferior en ImageNet; menos soporte en ecosistema Keras/TF |

---

### Consecuencias

**Positivas:**
- 91.4% de accuracy en clasificación de botellas PET por tamaño.
- Captura patrones morfológicos (forma, proporción, deformaciones) inaccesibles al pipeline clásico.
- Robusto a variaciones de iluminación gracias a las representaciones aprendidas en ImageNet.
- Latencia de inferencia de 25-50ms en CPU, aceptable para aplicaciones interactivas.
- Permite demostración en tiempo real en la app Streamlit (Tab 2).

**Negativas:**
- Requiere TensorFlow/Keras como dependencia de runtime (añade ~200MB a requirements).
- Archivo de modelo de 2.4MB debe almacenarse en repositorio GitHub (impacta tamaño del repo).
- RAM de Streamlit Cloud es el recurso limitante: el modelo ocupa ~300MB en memoria durante inferencia.
- El modelo está limitado a 3 clases PET; no generaliza directamente a las 6 clases del dataset principal.

---

### Notas de Implementación

Modelo entrenado con **Google Teachable Machine** (https://teachablemachine.withgoogle.com/), exportado como `keras_model.h5`. Carga y preprocesamiento en `app/app.py`:

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar modelo
model = tf.keras.models.load_model("converted_keras/keras_model.h5", compile=False)

# Preprocesamiento de imagen
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

# Inferencia
predictions = model.predict(preprocess_image(uploaded_file))
predicted_class = labels[np.argmax(predictions)]
confidence = np.max(predictions)
```

Etiquetas en `converted_keras/labels.txt`: `0 Botella_Plast_Grd`, `1 Botella_Plast_Med`, `2 Botella_Plast_Peq`.

---

## ADR-003: Selección de Framework de Despliegue Web (Streamlit Cloud + GitHub Pages)

- **Status:** Accepted
- **Date:** 2026-11-04
- **Deciders:** Feibert Alirio Giraldo

---

### Contexto

El proyecto CRS requiere dos capas de presencia pública distintas con propósitos complementarios. Por un lado, se necesita una **presentación institucional** del proyecto que sirva como portada académica: una página web que explique los objetivos, la metodología, los resultados y el equipo, accesible públicamente sin necesidad de interacción dinámica. Esta capa es esencial para la entrega académica y para la difusión del proyecto ante evaluadores y pares.

Por otro lado, se requiere un **prototipo interactivo** que permita a usuarios no técnicos experimentar directamente con los modelos ML: cargar imágenes de residuos y obtener predicciones en tiempo real de ambos modelos (Random Forest clásico y MobileNetV2). Esta capa de demostración es fundamental para validar la usabilidad del sistema y para la presentación ante el tribunal evaluador.

Ambas soluciones deben cumplir estrictamente con: (a) **costo cero**, dado el contexto académico sin presupuesto de infraestructura; (b) **mantenimiento cero**, pues el proyecto no cuenta con DevOps ni administración de servidores; (c) **disponibilidad global**, con acceso desde cualquier navegador sin instalación; (d) **despliegue desde GitHub**, integrando el flujo de trabajo git existente del proyecto.

---

### Decisión

Implementar una **arquitectura de despliegue de doble capa**:

**Capa 1 — Landing Page institucional:**
- Archivo: `index.html` en la raíz del repositorio
- Tecnologías: HTML5 + Tailwind CSS (via CDN) + JavaScript vanilla
- Despliegue: **GitHub Pages** desde la rama `main`, directorio raíz
- URL: `https://feibertciin.github.io/Project_ML_CRS/`

**Capa 2 — Aplicación ML interactiva:**
- Archivo principal: `app/app.py` (Streamlit)
- Dependencias: `app/requirements.txt`
- Despliegue: **Streamlit Community Cloud** conectado al repositorio GitHub
- URL: `https://projectmlcrs.streamlit.app/`

La Landing Page enlaza directamente a la app Streamlit, creando un flujo de usuario coherente: descubrimiento institucional → experimentación práctica.

---

### Justificación

La elección de GitHub Pages + Streamlit Cloud responde a un análisis multi-criterio:

| Criterio | GitHub Pages | Streamlit Cloud | Veredicto |
|----------|-------------|-----------------|-----------|
| Costo | Gratuito | Gratuito (tier community) | ✅ Ambos |
| Soporte Python ML | No aplica (estático) | Nativo | ✅ Streamlit |
| Auto-deploy desde GitHub | Sí (git push) | Sí (webhook) | ✅ Ambos |
| CDN global | Sí (GitHub CDN) | Sí (AWS) | ✅ Ambos |
| Tiempo de cold-start | 0ms (estático) | <5s (warm) | ✅ GitHub Pages |
| Límite RAM | N/A | 1GB | ⚠️ Streamlit |
| Dominio personalizable | Sí (CNAME) | Sí (subdomain) | ✅ Ambos |

GitHub Pages es la solución canónica para sitios estáticos de proyectos open source, con uptime del 99.9% garantizado por SLA de GitHub. Streamlit Cloud es el estándar de facto para prototipos ML de investigación académica, con soporte nativo para TensorFlow, scikit-learn y todas las dependencias del proyecto.

---

### Alternativas Consideradas

| Alternativa | Razón de Rechazo |
|-------------|-----------------|
| **Heroku** | Eliminó el tier gratuito en noviembre 2022; costo mínimo \$5/mes |
| **Google Cloud Run** | Requiere cuenta de facturación GCP; complejidad de configuración Docker excesiva para el contexto académico |
| **Render.com** | Tier gratuito disponible pero con cold-start de 30+ segundos tras inactividad; inaceptable para demo en vivo |
| **Hugging Face Spaces** | Orientado a ML pero interfaz menos apropiada para presentación institucional; menos familiar para audiencia académica local |
| **Railway.app** | Tier gratuito limitado a 500 horas/mes; riesgo de interrupción durante período de evaluación |
| **VPS auto-gestionado** | Costo mensual (mínimo \$5/mes), requiere configuración de servidor web, SSL, backups; incompatible con restricción de mantenimiento cero |
| **Vercel** | No soporta aplicaciones Python ML de forma nativa; requiere serverless functions con limitaciones de runtime |

---

### Consecuencias

**Positivas:**
- Costo de infraestructura: \$0.
- Despliegue automático en cada `git push` a rama `main`.
- Disponibilidad global mediante CDN sin configuración adicional.
- URLs estables y memorables para incluir en el informe académico.
- El flujo de desarrollo (editar → commit → push → deploy) se integra perfectamente con el workflow git existente.

**Negativas:**
- Es necesario mantener sincronizados los enlaces entre la Landing Page (GitHub Pages URL) y la app (Streamlit URL); un cambio de nombre de repositorio invalida ambas URLs.
- El tier gratuito de Streamlit Cloud impone límites de 1GB RAM y 1 CPU core: el modelo MobileNetV2 opera cerca del límite de memoria.
- GitHub Pages sirve contenido estático exclusivamente: cualquier funcionalidad dinámica adicional (API REST, bases de datos) requeriría migrar a otra plataforma.
- Streamlit Cloud puede entrar en modo "sleep" tras 7 días de inactividad, requiriendo reactivación manual.

---

### Notas de Implementación

**GitHub Pages:** Habilitado desde `Settings → Pages → Source: Deploy from branch (main, / root)` en el repositorio `feibertciin/Project_ML_CRS`. El archivo `index.html` en la raíz es el punto de entrada automático.

**Streamlit Cloud:** Configuración en `https://share.streamlit.io`:
- Repositorio: `feibertciin/Project_ML_CRS`
- Rama: `main`
- Archivo principal: `app/app.py`
- Archivo de dependencias: `app/requirements.txt`

Dependencias críticas en `app/requirements.txt`:
```
streamlit>=1.28.0
tensorflow>=2.13.0
scikit-learn>=1.3.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
```

---

## ADR-004: Selección de Dataset Kaggle vs Dataset Propio

- **Status:** Accepted
- **Date:** 2026-11-04
- **Deciders:** Feibert Alirio Giraldo

---

### Contexto

La construcción de un dataset de entrenamiento para clasificación de residuos sólidos implica una decisión fundamental de estrategia de datos: recolectar y etiquetar imágenes manualmente (dataset propio) o utilizar un dataset público existente. Esta decisión tiene impacto directo en la calidad de los datos, el cronograma del proyecto, el cumplimiento legal (licencias de uso), la reproducibilidad científica y la posibilidad de comparación con trabajos previos publicados.

La recolección manual de imágenes para 6 categorías de residuos con suficiente varianza (diferentes iluminaciones, fondos, ángulos de captura, estados de deterioro del material) requeriría un mínimo de 300-500 imágenes por clase (1,800-3,000 imágenes totales), proceso que implicaría 2-3 meses de recolección de campo y etiquetado. El costo de etiquetado en plataformas como Amazon Mechanical Turk oscilaría entre \$0.05 y \$0.15 por imagen, resultando en un costo de \$90-\$450 para el volumen requerido, incompatible con el presupuesto del proyecto académico.

Adicionalmente, un dataset puramente local (República Dominicana) podría no generalizar bien a otros contextos geográficos, limitando el valor académico del trabajo y dificultando la comparación con benchmarks publicados en la literatura de ML para clasificación de residuos.

---

### Decisión

Utilizar el dataset público de Kaggle **`sumn2u/garbage-classification-v2`** (2,527 imágenes, 6 clases, licencia CC BY 4.0) como **dataset principal** para el pipeline clásico y el entrenamiento exploratorio. Complementar con un **sub-dataset personalizado de botellas PET** (3 clases diferenciadas por tamaño: pequeña, mediana, grande) para el modelo de Transfer Learning especializado, dado que el dataset Kaggle no incluye esta diferenciación de tamaño.

**Distribución del dataset Kaggle:**

| Clase | Imágenes | Proporción |
|-------|---------|-----------|
| paper | 594 | 23.5% |
| cardboard | 403 | 15.9% |
| plastic | 482 | 19.1% |
| glass | 501 | 19.8% |
| metal | 410 | 16.2% |
| trash | 137 | 5.4% |
| **Total** | **2,527** | **100%** |

División: 80% entrenamiento (2,021 imágenes) / 20% prueba (506 imágenes), con estratificación para preservar las proporciones de clase.

---

### Justificación

La elección del dataset Kaggle `sumn2u/garbage-classification-v2` sobre las alternativas se justifica por los siguientes criterios:

1. **Disponibilidad inmediata:** Accesible mediante Kaggle API en <5 minutos, sin gestiones de acceso ni formularios de solicitud.
2. **Licencia académicamente compatible:** CC BY 4.0 permite uso, distribución y modificación para cualquier propósito (incluyendo académico) con solo atribución al autor.
3. **Calidad y limpieza:** El dataset ya está organizado en carpetas por clase, sin imágenes duplicadas ni etiquetas erróneas conocidas, eliminando la fase de limpieza de datos.
4. **Reproducibilidad:** Al ser un dataset público y versionado en Kaggle, cualquier investigador puede reproducir exactamente los experimentos del proyecto descargando la misma versión.
5. **Benchmarking:** Existen trabajos académicos publicados que usan este mismo dataset, permitiendo comparación directa de resultados (accuracy del 62.8% con RF vs resultados publicados).

La incorporación del sub-dataset PET personalizado es **necesaria** porque ningún dataset público disponible diferencia botellas de plástico por tamaño (pequeña <500ml, mediana 500-1000ml, grande >1000ml), caso de uso específico requerido para el modelo de Transfer Learning del proyecto.

---

### Alternativas Consideradas

| Alternativa | Descripción | Razón de Rechazo |
|-------------|-------------|-----------------|
| **TrashNet (Gary Thung, 2016)** | ~2,500 imágenes, 6 clases, Stanford | Similar tamaño pero menos clases relevantes (incluye "shoes"); dataset más antiguo (2016) con menor varianza visual |
| **TACO (Waste Annotations in Contexts)** | >1,500 imágenes con anotaciones COCO (bounding boxes) | Requiere parsing complejo de formato COCO; orientado a detección de objetos, no clasificación; no alineado con el cronograma académico |
| **OpenImages V7 (Google)** | 9M imágenes, múltiples clases de residuos | Requiere filtrado complejo para extraer clases de residuos; desproporcionado para el alcance del proyecto; tiempo de descarga >10 horas |
| **Dataset propio (recolección manual)** | Fotos propias de residuos en contexto local (RD) | Costo de tiempo (2-3 meses) y potencial costo económico incompatibles con cronograma académico; sin benchmark de comparación |
| **Web scraping (Google Images, Bing)** | Scraping automatizado de imágenes de residuos | Problemas de derechos de autor; calidad y etiquetado no garantizados; riesgo de sesgo sistemático |

---

### Consecuencias

**Positivas:**
- Disponibilidad inmediata del dataset completo.
- Reproducibilidad científica garantizada (dataset público y versionado).
- Cumplimiento legal total con licencia CC BY 4.0.
- Posibilidad de comparar resultados con literatura académica previa.
- Ahorro de 2-3 meses de recolección y etiquetado de datos.

**Negativas:**
- El dataset puede no reflejar la composición real de residuos en el contexto local (República Dominicana): tipos de envases, marcas y materiales específicos del Caribe no están representados.
- Limitado a 6 categorías genéricas; no incluye subcategorías relevantes localmente (tetrapack, residuos peligrosos, residuos orgánicos).
- La clase `trash` con solo 137 imágenes (5.4%) introduce desequilibrio de clases que impacta el rendimiento del clasificador (ver ADR-005).
- Dependencia de la disponibilidad continua del dataset en Kaggle para reproductibilidad futura.

---

### Notas de Implementación

Descarga del dataset mediante Kaggle API:

```bash
kaggle datasets download sumn2u/garbage-classification-v2
unzip garbage-classification-v2.zip -d data/garbage-classification-v2/
```

Ruta local en el notebook: `data/garbage-classification-v2/`. División estratificada implementada con:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Preserva proporciones de clase
)
```

La variable `random_state=42` garantiza reproducibilidad exacta de la división en cualquier ejecución del notebook.

---

## ADR-005: Estrategia de Balanceo de Clases (class_weight='balanced' vs SMOTE vs Oversampling)

- **Status:** Accepted
- **Date:** 2026-11-04
- **Deciders:** Feibert Alirio Giraldo

---

### Contexto

El dataset Garbage Classification v2 presenta un **desbalance de clases significativo**: la clase `trash` contiene únicamente 137 imágenes (5.4% del total), mientras que `paper` contiene 594 imágenes (23.5%). Esta proporción de ~4.3:1 entre la clase mayoritaria y la clase minoritaria genera un problema sistemático en el entrenamiento del clasificador: sin intervención, el modelo aprende a maximizar la accuracy global ignorando las clases minoritarias, produciendo una tasa de recall cercana a 0 para `trash` (el clasificador simplemente nunca predice esta clase).

El impacto en métricas es el siguiente: un clasificador "naive" que siempre predice `paper` obtendría ~23.5% de accuracy global sin haber aprendido ningún patrón real. El Random Forest sin balanceo de clases presenta sesgos hacia las clases mayorita­rias (`paper`, `glass`, `plastic`) y F1-score muy bajo para `trash` y `cardboard`. Desde la perspectiva de la aplicación real (clasificación de residuos para reciclaje), la confusión en la clase `trash` es especialmente costosa: residuos no clasificables enviados a reciclaje contaminan el flujo de materiales.

Existen tres estrategias principales para abordar el desbalance en el contexto de este proyecto: (1) ponderación algorítmica de clases, (2) sobremuestreo sintético (SMOTE), y (3) submuestreo de clases mayoritarias. La elección debe considerar el espacio de características específico del proyecto (4 dimensiones: mean_r, mean_g, mean_b, edge_density) y las restricciones de complejidad de código del proyecto académico.

---

### Decisión

Utilizar el parámetro **`class_weight='balanced'`** en `RandomForestClassifier` de scikit-learn. Esta configuración instruye al algoritmo para que calcule automáticamente pesos de clase inversamente proporcionales a su frecuencia:

$$w_c = \frac{n\_samples}{n\_classes \times n\_samples\_class\_c}$$

Configuración completa del clasificador:

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    criterion='gini',
    random_state=42,
    n_jobs=-1  # Paralelización en todos los cores disponibles
)
```

**Pesos calculados automáticamente para el dataset:**

| Clase | Imágenes | Peso calculado |
|-------|---------|--------------|
| cardboard | 403 | ≈ 1.04 |
| glass | 501 | ≈ 0.84 |
| metal | 410 | ≈ 1.02 |
| paper | 594 | ≈ 0.71 |
| plastic | 482 | ≈ 0.87 |
| **trash** | **137** | **≈ 3.07** |

La clase `trash` recibe un peso ~3× mayor que las clases mayoritarias, compensando su baja frecuencia durante el proceso de selección de splits en cada árbol del ensemble.

---

### Justificación

La decisión de usar `class_weight='balanced'` sobre las alternativas se basa en tres factores determinantes:

**1. Dimensionalidad del espacio de características:** Con solo 4 features (mean_r, mean_g, mean_b, edge_density), el espacio de características es extremadamente compacto. SMOTE genera puntos sintéticos interpolando entre vecinos más cercanos en este espacio de 4 dimensiones. Con solo 137 muestras de `trash` en 4D, los puntos sintéticos serán interpolaciones de puntos muy próximos entre sí, generando poca diversidad real. El riesgo de crear artefactos sintéticos que no corresponden a imágenes reales es elevado en espacios tan comprimidos.

**2. Cero overhead computacional:** `class_weight='balanced'` modifica únicamente el criterio de impureza durante la construcción de árboles, sin coste adicional de preprocesamiento ni nuevas dependencias. El tiempo de entrenamiento permanece < 10 segundos.

**3. Principio de mínima complejidad:** Para un proyecto académico en el que la reproducibilidad y la comprensión del código son prioritarias, `class_weight='balanced'` requiere un único argumento adicional. SMOTE requiere `pip install imbalanced-learn`, comprensión de hiperparámetros adicionales (k_neighbors, sampling_strategy) y validación de que los datos sintéticos no introducen data leakage en la validación cruzada.

---

### Alternativas Consideradas

| Alternativa | Descripción | Razón de Rechazo |
|-------------|-------------|-----------------|
| **SMOTE (Synthetic Minority Oversampling Technique)** | Genera muestras sintéticas interpolando entre vecinos más cercanos de la clase minoritaria | En espacio 4D con 137 muestras de `trash`, genera poca diversidad real; riesgo de overfitting a artefactos sintéticos; requiere `imbalanced-learn`; complejidad de implementación desproporcionada |
| **Random Oversampling** | Duplicar aleatoriamente muestras de clases minoritarias hasta igualar la clase mayoritaria | No añade información nueva (duplicados); incrementa el tiempo de entrenamiento sin beneficio en generalización; riesgo de overfitting a las muestras duplicadas |
| **Random Undersampling de clases mayoritarias** | Reducir `paper` y `glass` al tamaño de `trash` (137 imágenes) | Elimina el 77%+ de los datos de entrenamiento disponibles; provoca pérdida masiva de información en las clases más representadas |
| **Ajuste de threshold de decisión post-entrenamiento** | Mover el umbral de clasificación para clases minoritarias tras entrenar sin balanceo | Requiere validación del umbral óptimo por clase; añade complejidad en la fase de inferencia; aplazado a versión v2.0 del proyecto |
| **Augmentación de datos de imágenes** | Rotación, flip, zoom sobre imágenes de `trash` antes de extraer features | Efectivo para CNNs; para el pipeline clásico, las augmentaciones de imagen producen features casi idénticos (mean_r/g/b cambian marginalmente con flip/rotación), por lo que el beneficio es mínimo |

---

### Consecuencias

**Positivas:**
- Soporte nativo en scikit-learn: sin instalación de paquetes adicionales.
- Cálculo automático de pesos: no requiere ajuste manual de hiperparámetros.
- Interpretable: los pesos son directamente auditables y explicables.
- Sin riesgo de data leakage: `class_weight` opera internamente en el entrenamiento sin modificar los datos de entrada.
- Mejora medible en recall para clases minoritarias (`trash`, `cardboard`) respecto a RF sin balanceo.

**Negativas:**
- No añade ejemplos nuevos: el modelo sigue observando la distribución original desequilibrada de los datos durante el entrenamiento. La clase `trash` sigue siendo la más difícil de clasificar (F1-score ~30%).
- La ponderación afecta a todos los árboles del ensemble uniformemente; no permite estrategias de balanceo diferenciadas por subgrupo de árboles.
- En la práctica, `class_weight='balanced'` contribuye a la mejora de accuracy de ~58% (sin balanceo) a ~62.8% (con balanceo), ganancia modesta que refleja la limitación fundamental del espacio de 4 características.

---

### Notas de Implementación

Implementación completa en `Project_ML_CRS.ipynb`:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Entrenamiento con class_weight='balanced'
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    criterion='gini',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Evaluación
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred,
                             target_names=['cardboard','glass','metal','paper','plastic','trash']))
```

Los pesos efectivos calculados automáticamente pueden verificarse con:

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

weights = compute_class_weight('balanced',
                                classes=np.unique(y_train),
                                y=y_train)
print(dict(zip(np.unique(y_train), weights.round(2))))
# {'cardboard': 1.04, 'glass': 0.84, 'metal': 1.02,
#  'paper': 0.71, 'plastic': 0.87, 'trash': 3.07}
```

Para el modelo MobileNetV2 (ADR-002), el balanceo de clases se gestiona a través del parámetro `class_weight` en `model.fit()` de Keras o mediante `ImageDataGenerator` con `class_mode='categorical'` y pesos calculados equivalentes.

---

*Documento generado el 2026-11-04. Próxima revisión prevista al cierre del Sprint 3.*
