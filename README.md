# Clasificación de Animales con Redes Neuronales Convolucionales

Trabajo de Fin de Grado — Implementación y comparativa de arquitecturas CNN para la clasificación automática de imágenes de **6 especies animales**: mariposas, gallinas, elefantes, caballos, arañas y ardillas.

---

## Descripción

Este proyecto estudia el impacto de distintas técnicas de diseño y regularización sobre una CNN entrenada desde cero, partiendo de un modelo baseline (red neuronal clásica sin capas convolucionales) y escalando progresivamente en complejidad hasta un modelo final que combina Batch Normalization, Global Average Pooling y Data Augmentation. Se incluye también una comparativa con Transfer Learning mediante EfficientNetB0 preentrenada en ImageNet.

---

## Estructura del proyecto

```
proyecto/
├── dataset/                  ← submódulo Git (ver instrucciones abajo)
│   ├── train/
│   ├── validation/
│   └── test/
├── utils/
│   ├── __init__.py
│   ├── dataset_loader.py     ← carga y preprocesado del dataset
│   ├── plot_training.py      ← visualización de curvas de aprendizaje
│   └── metrics.py            ← evaluación, matriz de confusión e informe
├── models/
│   ├── 01_baseline_ann.py    ← Red neuronal clásica (sin convolución)
│   ├── 02_cnn_basic.py       ← CNN básica (3 bloques + Flatten)
│   ├── 03_cnn_data_augmentation.py
│   ├── 04_cnn_class_weights.py
│   ├── 05a_cnn_no_dropout.py ← Experimento: sin Dropout (overfitting)
│   ├── 05b_cnn_extreme_dropout.py ← Experimento: Dropout 0.9 (underfitting)
│   ├── 06_cnn_batch_normalization.py
│   ├── 07_cnn_gap.py         ← Global Average Pooling
│   ├── 08_cnn_deep.py        ← CNN profunda (4 bloques)
│   ├── 09_cnn_deep_bn.py     ← CNN profunda + Batch Normalization
│   ├── 10_cnn_final.py       ← Modelo final (4 bloques + GAP + BN + DA)
│   └── 11_efficientnet.py    ← Transfer Learning con EfficientNetB0
└── results/
    ├── plots/                ← curvas de aprendizaje y matrices de confusión
    └── reports/              ← informes de clasificación por modelo
```

---

## Dataset

El dataset contiene **13.412 imágenes de entrenamiento**, 2.549 de validación y 1.845 de test, distribuidas en 6 clases con desbalanceo notable (arañas ~30%, elefantes <10%).

El dataset está disponible como submódulo Git en:
👉 `https://github.com/[usuario]/[repo-dataset]`

### Opción 1 — Clonar con submódulo (recomendado)

```bash
git clone --recurse-submodules https://github.com/tu-usuario/tu-repo
```

### Opción 2 — Si ya tienes el repo clonado

```bash
git submodule update --init
```

### Opción 3 — Clonar el dataset manualmente

Si el submódulo no carga correctamente, clona el dataset directamente:

```bash
git clone https://github.com/[usuario]/[repo-dataset] dataset
```

---

## Instalación

### Requisitos

- Python 3.9+
- TensorFlow 2.x / Keras
- NumPy, Matplotlib, scikit-learn

### Instalar dependencias

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

## Uso

Todos los scripts se ejecutan **desde la carpeta raíz del proyecto**, no desde dentro de `models/`.

```bash
# Verificar que estás en la raíz
pwd  # debe mostrar .../tu-proyecto/

# Ejecutar un modelo
python models/02_cnn_basic.py
```

Los resultados (gráficas e informes) se guardan automáticamente en `results/plots/` y `results/reports/`.

---

## Modelos implementados

| # | Modelo | Accuracy | Observaciones |
|---|--------|----------|---------------|
| 01 | Baseline ANN | 56,42% | Sin capas convolucionales |
| 02 | CNN Básica | 73,28% | 3 bloques + Flatten |
| 03 | CNN + Data Augmentation | ~82% | Elimina overfitting |
| 04 | CNN + Class Weights | 76,15% | Mejora clases minoritarias |
| 05a | CNN sin Dropout | 73,28% | Overfitting masivo |
| 05b | CNN Dropout 0.9 | 58,97% | Underfitting severo |
| 06 | CNN + Batch Normalization | ~74% | Estabiliza entrenamiento |
| 07 | CNN + GAP | ~80% | Reduce parámetros drásticamente |
| 08 | CNN Profunda (4 bloques) | ~80% | Mayor capacidad |
| 09 | CNN Profunda + BN | ~84% | Resuelve desbalanceo |
| 10 | Modelo Final | ~88% | 4 bloques + GAP + BN + DA |
| 11 | EfficientNetB0 | ~98% | Transfer Learning |

---

## Métricas de evaluación

Se utiliza **F1-score macro** como métrica principal por ser la más adecuada para datasets con desbalanceo de clases. Se complementa con matriz de confusión y accuracy global como referencia secundaria.

---

## Autor

Trabajo de Fin de Grado — [Javier Torres López]  
[UNED] — [2026]
