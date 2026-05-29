#!/usr/bin/env python
# coding: utf-8

"""
models/01_baseline_ann.py
--------------------------
Red Neuronal Artificial clásica (fully connected) sin capas convolucionales.
Sirve como modelo baseline y umbral mínimo de comparación.

Arquitectura:
    Input (299x299x3) → Resizing (32x32) → Rescaling → Flatten
    → Dense(1024, ReLU) → Dropout(0.3)
    → Dense(256, ReLU)  → Dropout(0.2)
    → Dense(6, Softmax)
"""

# ============================================================
# IMPORTACIONES
# ============================================================
import tensorflow as tf
import numpy as np
import keras
import time
import pickle
from keras.layers import Flatten, Input, Dense, Dropout, Rescaling, Resizing
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from tensorflow.keras.models import load_model

from utils.dataset_loader import load_datasets
from utils.plot_training import plot_training, FakeHistory
from utils.metrics import evaluate_model

# ============================================================
# CARGA DEL DATASET
# ============================================================
train_ds, val_ds, test_ds = load_datasets()

# ============================================================
# DEFINICIÓN DEL MODELO
# ============================================================
input_layer = keras.Input(shape=(299, 299, 3), name="image")

x = Resizing(32, 32)(input_layer)
x = Rescaling(1./255)(x)
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(6, activation="softmax")(x)

ann_model = keras.Model(input_layer, x, name="baseline_ann")
ann_model.summary()

# ============================================================
# COMPILACIÓN
# ============================================================
opt = Adam(learning_rate=0.0001)

ann_model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

# ============================================================
# ENTRENAMIENTO
# ============================================================
t = time.time()
ann_history = ann_model.fit(
    train_ds,
    epochs=100,
    callbacks=[stopping],
    verbose=1,
    validation_data=val_ds
)
print("Tiempo de entrenamiento: ", time.time() - t, "segundos.")

# Guardado del modelo e historial
ann_model.save('modelo_baseline_ann.keras')
print("Modelo guardado en el disco.")

with open('historial_baseline_ann.pkl', 'wb') as f:
    pickle.dump(ann_history.history, f)
print("Historial guardado en el disco.")

# ============================================================
# RESULTADOS
# ============================================================
with open('historial_baseline_ann.pkl', 'rb') as f:
    datos = pickle.load(f)

ann_history = FakeHistory(datos)
n_epochs = len(ann_history.history["loss"])
plot_training(n_epochs, ann_history, save_path='results/plots/01_baseline_ann.png')

# Cargo el modelo y evalúo
ann_model = load_model('modelo_baseline_ann.keras')
evaluate_model(ann_model, test_ds,
               save_path='results/plots/01_baseline_ann_confusion.png',
               report_path='results/reports/01_baseline_ann_report.txt',
               model_name='Baseline ANN')
