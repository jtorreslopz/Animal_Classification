#!/usr/bin/env python
# coding: utf-8

"""
models/03_cnn_data_augmentation.py
------------------------------------
CNN básica con Data Augmentation para combatir el desbalanceo de clases
y el overfitting mediante transformaciones aleatorias en el preprocesado.

Transformaciones aplicadas:
    - RandomFlip horizontal
    - RandomRotation (±10%)
    - RandomZoom (±10%)
    - RandomContrast (±10%)

Arquitectura: igual que CNN básico con Data Augmentation antes del extractor.
"""

# ============================================================
# IMPORTACIONES
# ============================================================
import tensorflow as tf
import numpy as np
import os
import keras
import time
import pickle
from keras.layers import (Flatten, Input, Dense, Dropout,
                           Conv2D, MaxPooling2D, Rescaling,
                           RandomFlip, RandomRotation, RandomZoom, RandomContrast)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from utils.dataset_loader import load_datasets
from utils.plot_training import plot_training, FakeHistory
from utils.metrics import evaluate_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# CARGA DEL DATASET
# ============================================================
train_ds, val_ds, test_ds = load_datasets()

# ============================================================
# DATA AUGMENTATION
# ============================================================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# ============================================================
# DEFINICIÓN DEL MODELO
# ============================================================
cnn_model_augmentation = Sequential()

# Preprocesado + Augmentation
cnn_model_augmentation.add(Input(shape=(299, 299, 3)))
cnn_model_augmentation.add(Rescaling(1./255))
cnn_model_augmentation.add(data_augmentation)

# Bloque 1: Bordes y colores
cnn_model_augmentation.add(Conv2D(16, 3, padding='same', activation='relu'))
cnn_model_augmentation.add(MaxPooling2D())

# Bloque 2: Texturas
cnn_model_augmentation.add(Conv2D(32, 3, padding='same', activation='relu'))
cnn_model_augmentation.add(MaxPooling2D())

# Bloque 3: Formas complejas
cnn_model_augmentation.add(Conv2D(64, 3, padding='same', activation='relu'))
cnn_model_augmentation.add(MaxPooling2D())
cnn_model_augmentation.add(Dropout(0.2))

# Clasificador
cnn_model_augmentation.add(Flatten())
cnn_model_augmentation.add(Dense(128, activation="relu"))
cnn_model_augmentation.add(Dropout(0.3))
cnn_model_augmentation.add(Dense(6, name="outputs", activation="softmax"))

cnn_model_augmentation.summary()

# ============================================================
# COMPILACIÓN
# ============================================================
opt = Adam(learning_rate=0.001)

cnn_model_augmentation.compile(
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

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)

# ============================================================
# ENTRENAMIENTO
# ============================================================
t = time.time()
cnn_augmentation_history = cnn_model_augmentation.fit(
    train_ds,
    epochs=100,
    callbacks=[stopping, reduce_lr],
    verbose=1,
    validation_data=val_ds
)
print("Tiempo de entrenamiento: ", time.time() - t, "segundos.")

# Guardado del modelo e historial
cnn_model_augmentation.save('modelo_cnn_augmentation.keras')
print("Modelo guardado en el disco.")

with open('historial_cnn_augmentation.pkl', 'wb') as f:
    pickle.dump(cnn_augmentation_history.history, f)
print("Historial guardado en el disco.")

# ============================================================
# RESULTADOS
# ============================================================
with open('historial_cnn_augmentation.pkl', 'rb') as f:
    datos = pickle.load(f)

cnn_augmentation_history = FakeHistory(datos)
n_epochs = len(cnn_augmentation_history.history["loss"])
plot_training(n_epochs, cnn_augmentation_history,
              save_path='results/plots/03_cnn_data_augmentation.png')

# Cargo el modelo y evalúo
cnn_model_augmentation = load_model('modelo_cnn_augmentation.keras')
evaluate_model(cnn_model_augmentation, test_ds,
               save_path='results/plots/03_cnn_data_augmentation_confusion.png',
               report_path='results/reports/03_cnn_data_augmentation_report.txt',
               model_name='CNN + Data Augmentation')
