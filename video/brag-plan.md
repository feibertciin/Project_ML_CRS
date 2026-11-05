# Brag Plan: Clasificación Inteligente de Residuos Sólidos (CRS)

## What is this app?
Sistema dual de Machine Learning y Deep Learning que clasifica automáticamente residuos sólidos urbanos con un pipeline ETL clásico (Random Forest) y Transfer Learning (MobileNetV2), desplegado como app interactiva en Streamlit con 91.4% de precisión.

## The angle
"Convirtiendo 2,527 imágenes de basura en analítica predictiva — sin GPU, en 5 milisegundos."
El gancho: la paradoja de que un modelo liviano (4 variables) logra clasificar residuos mientras un modelo profundo (1280 features) los discrimina con alta precisión — y ambos corren juntos en Streamlit.

## Hook (first 2-3 seconds)
Texto grande, blanco sobre fondo oscuro: "2,527 imágenes. 4 variables. 6 categorías de residuos." — rápido, directo, impactante.

## Key moments
- El simulador Streamlit con sliders de color RGB y edge_density: selección en vivo, barra de probabilidades
- El comparativo en dos pestañas: Random Forest vs MobileNetV2 — la dualidad del proyecto
- Métricas reales animadas: 62.8% RF → 91.4% Transfer Learning — la mejora visible

## Outro / punchline
"Código abierto. Desplegado en la nube. Listo para economía circular." + Badge verde UFHEC

## User flow worth showing
1. Entrada: Deslizadores de intensidad espectral (mean_r, mean_g, mean_b, edge_density)
2. Acción clave: Predicción instantánea con barra de confianza por clase
3. Resultado: Comparación Random Forest vs MobileNetV2 en dos pestañas

## Tone
- Preset: polished
- Creative direction: Académico-profesional con energía de startup tech verde
- Interpretation: Ritmo moderado, transiciones suaves cross-fade, tipografía display grande, colores esmeralda/marino

## Format: landscape — 1920x1080
## Duration: 20 seconds

## Visual identity
- Background: #0b1120
- Accent: #10b981
- Text: #f1f5f9
- Display font: Outfit
- Body font: Inter
- Strongest visual element: Las dos pestañas comparativas de Streamlit con gráficos de barras

## Share copy (draft)
"Automatizamos la clasificación de residuos sólidos con ML clásico y Transfer Learning — código abierto, 91.4% de precisión, 5ms de inferencia."

## Audio direction
- Role: warm bed with professional energy
- Music: happy-beats-business-moves-vol-9-by-ende-dot-app.mp3 (114.84 BPM)
- Music treatment: fade-in 0.5s, volume 0.25 (ducked para no tapar contenido visual), fade-out 1s al final
- Music cue guidance: strongCues en 4.23s, 6.34s, 10.54s, 12.65s — alinear revelaciones principales
- Audio-reactive treatment: subtle — fondo/glow reacciona suavemente al RMS
- SFX posture: sparse — clicks en transiciones de pestañas, card reveal en métricas
- Restraint rule: No waveform bars, no musical-note graphics

## Storyboard

### Scene 1 — Hook Estadístico — 3s
Fondo #0b1120. Texto centrado blanco grande: "2,527 imágenes → 4 variables → 6 categorías"
Los 3 elementos aparecen secuencialmente con ritmo.
Sequential/interaction: sí — cada elemento aparece uno por uno con 0.8s de intervalo
Audio intent: establecer ritmo y energía inicial
Audio-coupled idea: cada aparición de texto sincronizada con beat (3.70s, 4.23s, 4.75s del grid)
Music: upbeat business
Transition mood: clean → Scene 2

### Scene 2 — Logo y Título Proyecto — 3s
Icono ♻️ grande + título "Clasificación Inteligente de Residuos Sólidos" + badge "Maestría IA — UFHEC"
Fade in elegante desde negro, escalado ligero
Sequential/interaction: none
Audio intent: profesional, establecer identidad
Audio-coupled idea: beat-locked al strongCue 6.34s
Music: warm corporate
Transition mood: soft → Scene 3

### Scene 3 — Pipeline ETL en Acción — 5s
Recreación del flujo: imagen → extracción RGB+Canny → DataFrame → RandomForest → Clase predicha
Diagrama en 5 pasos que aparecen secuencialmente de izquierda a derecha con flechas
Sequential/interaction: sí — 5 pasos del pipeline aparecen uno por uno
Audio intent: demostrar proceso técnico con confianza
Audio-coupled idea: cada paso snap a beat grid (8.44s, 8.96s, 9.50s, 10.01s, 10.54s)
Music: continuous upbeat
Transition mood: clean → Scene 4

### Scene 4 — Comparativa de Modelos — 5s
Dos paneles lado a lado: "Random Forest — 62.8%" vs "MobileNetV2 — 91.4%"
Barras de precisión animadas llenándose
Sequential/interaction: sí — barras se llenan simultáneamente, luego texto de diferencia aparece
Audio intent: revelar el resultado del benchmark, climax del video
Audio-coupled idea: beat-locked al strongCue 12.65s para el reveal final
Music: swell energético
Transition mood: dramatic → Scene 5

### Scene 5 — CTA y Cierre — 4s
"Código Abierto. GitHub + Streamlit." con dos badges
URLs visibles: github.com/feibertciin/Project_ML_CRS | Project_ML_CRS.streamlit.app
Badge verde UFHEC centrado al final
Sequential/interaction: none
Audio intent: landing profesional, fade out
Audio-coupled idea: beat-locked al strongCue 23.17s para el badge final
Music: fade out elegante
Transition mood: none (fin)

**Music mood for this video:** upbeat professional
**Audio summary:** Track de negocios alegre a 0.25 de volumen con fade in/out, SFX de card reveal y clicks para las transiciones entre escenas y apariciones de métricas.
