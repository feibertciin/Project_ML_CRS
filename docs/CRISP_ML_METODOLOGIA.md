# Guía Metodológica: Ciclo de Vida CRISP-ML(Q)
## Proyecto Final: Clasificación Inteligente de Residuos Sólidos (CRS)
**Maestría en IA, Educación e Investigación | Asignatura: Machine Learning**

---

### Introducción a CRISP-ML(Q)
El estándar **CRISP-ML(Q)** (*Cross-Industry Standard Process for the development of Machine Learning applications with Quality QA*) extiende el tradicional CRISP-DM añadiendo aseguramiento de calidad, monitorización continua y mitigación de riesgos propios del software inteligente. Este documento rige todas las decisiones técnicas tomadas en el proyecto.

```
       ┌────────────────────────────────────────────────────────┐
       │     1. Comprensión del Negocio y del Problema           │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     2. Comprensión e Ingesta de Datos (ETL)             │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     3. Preparación de Datos e Ingeniería de Features    │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     4. Modelado y Entrenamiento (RF + MobileNetV2)      │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     5. Evaluación Técnica y Validación de Negocio      │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     6. Despliegue, Landing Page y Monitoreo (Streamlit) │
       └────────────────────────────────────────────────────────┘
```

---

### Fase 1: Comprensión del Negocio y del Problema (Business Understanding)
1. **Objetivo Estratégico:**
   - Mitigar el impacto ambiental de los residuos sólidos no segregados mediante un sistema computacional de bajo costo para clasificación automatizada.
2. **Criterios de Éxito:**
   - **Técnico:** Obtener un Accuracy superior al 60% en clasificación rápida multiclase tabular (Random Forest) y superior al 85% en clasificación fina de materiales reciclables críticos (botellas plásticas con Transfer Learning).
   - **Operacional:** Latencia de inferencia menor a 150 ms por imagen en entorno web Streamlit.
3. **Casos de Uso:**
   - Puntos limpios inteligentes.
   - Herramienta pedagógica gamificada para concienciación ciudadana.

---

### Fase 2: Comprensión y Adquisición de Datos (Data Understanding)
1. **Dataset Principal:** *Garbage Classification v2* (Kaggle: `sumn2u/garbage-classification-v2`).
2. **Estructura:** 6 clases (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`).
3. **Dataset Secundario (Transfer Learning):** Imágenes estructuradas de botellas plásticas organizadas por volumetría (`Botella_Plast_Peq`, `Botella_Plast_Med`, `Botella_Plast_Grd`).
4. **Análisis de Calidad:**
   - Inspección de variabilidad lumínica, fondos heterogéneos y oclusiones parciales.

---

### Fase 3: Preparación de Datos e Ingeniería de Características (Data Preparation)
1. **Pipeline ETL Visual-Tabular:**
   - Redimensionamiento estándar a 100x100 píxeles.
   - Descomposición en canales espectrales RGB: `mean_r`, `mean_g`, `mean_b` para capturar la signatura cromática del residuo.
   - Extracción de textura estructural: Algoritmo de detección de bordes de Canny (umbrales 50 y 150) y cálculo de densidad de bordes (`edge_density`).
2. **Preprocesamiento:**
   - Escalado estándar (`StandardScaler`) aplicado exclusivamente al conjunto de entrenamiento (`fit_transform`) y proyectado a validación (`transform`) para prevenir data leakage.
   - Partición estratificada 80/20.

---

### Fase 4: Modelado (Modeling)
Se implementa una arquitectura comparativa de dos paradigmas:
1. **Enfoque Clásico (Tabular / Features Manuales):**
   - **Random Forest Classifier:** 100 estimadores, balanceo de pesos de clase (`class_weight='balanced'`), criterio Gini.
   - **Ventajas:** Extremadamente ligero, ejecutable en CPU básica o microcontrolador, altamente interpretable.
2. **Enfoque Deep Learning (Transfer Learning):**
   - **MobileNetV2 preentrenado con ImageNet:** Capas convolucionales profundas congeladas, *Global Average Pooling* y clasificador denso ajustado con Dropout y Softmax.
   - **Ventajas:** Captura patrones morfológicos y de textura complejos sin ingeniería manual.

---

### Fase 5: Evaluación (Evaluation)
1. **Métricas Clave:**
   - Matriz de confusión para detectar confusiones cruzadas críticas (ej. papel vs cartón, o plástico transparente vs vidrio).
   - Precision, Recall y F1-Score ponderado por clase.
2. **Robustez y Límites:**
   - Sensibilidad a la iluminación: El modelo tabular es sensible a sombras extremas; MobileNetV2 ofrece mayor invariancia espacial gracias a las capas convolucionales intermedias.

---

### Fase 6: Despliegue y Operación (Deployment & Monitoring)
1. **Landing Page Web:** Arquitectura en capas (Header/Hero, Estadísticas, Pipeline CRISP-ML, Simulador/Demo, Comparativa de Modelos, Recursos y Enlaces a GitHub/Streamlit).
2. **Aplicación Web Interactiva Streamlit:**
   - Pestaña 1: Análisis Exploratorio, Extracción ETL y Simulador del modelo clásico Random Forest del notebook.
   - Pestaña 2: Inferencia en tiempo real (imagen o webcam) con el modelo de Transfer Learning (`keras_model.h5` y `labels.txt`).
3. **Mantenimiento:** Reentrenamiento programable ante *data drift* o nuevos tipos de empaques.
