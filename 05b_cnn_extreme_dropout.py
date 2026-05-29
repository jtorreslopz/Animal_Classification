#!/usr/bin/env python
# coding: utf-8

"""
models/05b_cnn_extreme_dropout.py
-----------------------------------
Experimento de control: CNN básica con Dropout agresivo del 90%.
Demuestra el underfitting severo causado por una regularización excesiva.

Arquitectura: igual que CNN básico pero con Dropout(0.9) en ambas posiciones.
"""

# ============================================================
# IMPORTACIONES
# ============================================================
import time
import pickle
from keras.layers import (Flatten, Input, Dense, Dropout,
                           Conv2D, MaxPooling2D, Rescaling)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import Sequential
from tensorflow.keras.models import load_model

from utils.dataset_loader import load_datasets
from utils.plot_training import plot_training, FakeHistory
from utils.metrics import evaluate_model

# ============================================================
# CARGA DEL DATASET
# ============================================================
train_ds, val_ds, test_ds = load_datasets()

# ============================================================
# DEFINICIÓN DEL MODELO — Dropout agresivo 0.9
# ============================================================
cnn_model_grandrop = Sequential()

# Extractor de características
cnn_model_grandrop.add(Input(shape=(299, 299, 3)))
cnn_model_grandrop.add(Rescaling(1./255))

# Bloque 1
cnn_model_grandrop.add(Conv2D(16, 3, padding='same', activation='relu'))
cnn_model_grandrop.add(MaxPooling2D())

# Bloque 2
cnn_model_grandrop.add(Conv2D(32, 3, padding='same', activation='relu'))
cnn_model_grandrop.add(MaxPooling2D())

# Bloque 3
cnn_model_grandrop.add(Conv2D(64, 3, padding='same', activation='relu'))
cnn_model_grandrop.add(MaxPooling2D())
cnn_model_grandrop.add(Dropout(0.9))  # ⚠️ Dropout agresivo en extractor

# Clasificador — Dropout agresivo
cnn_model_grandrop.add(Flatten())
cnn_model_grandrop.add(Dense(64, activation='relu'))
cnn_model_grandrop.add(Dropout(0.9))  # ⚠️ Dropout agresivo en clasificador
cnn_model_grandrop.add(Dense(6, name="outputs", activation="softmax"))

cnn_model_grandrop.summary()

# ============================================================
# COMPILACIÓN
# ============================================================
opt = Adam(learning_rate=0.001)

cnn_model_grandrop.compile(
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
cnn_history_grandrop = cnn_model_grandrop.fit(
    train_ds,
    epochs=100,
    callbacks=[stopping, reduce_lr],
    verbose=1,
    validation_data=val_ds
)
print("Tiempo de entrenamiento: ", time.time() - t, "segundos.")

# Guardado del modelo e historial
cnn_model_grandrop.save('cnn_model_grandrop.keras')
print("Modelo guardado en el disco.")

with open('cnn_history_grandrop.pkl', 'wb') as f:
    pickle.dump(cnn_history_grandrop.history, f)
print("Historial guardado en el disco.")

# ============================================================
# RESULTADOS
# ============================================================
with open('cnn_history_grandrop.pkl', 'rb') as f:
    datos = pickle.load(f)

cnn_history_grandrop = FakeHistory(datos)
n_epochs = len(cnn_history_grandrop.history["loss"])
plot_training(n_epochs, cnn_history_grandrop,
              save_path='results/plots/05b_cnn_extreme_dropout.png')

# Cargo el modelo y evalúo
cnn_model_grandrop = load_model('cnn_model_grandrop.keras')
evaluate_model(cnn_model_grandrop, test_ds,
               save_path='results/plots/05b_cnn_extreme_dropout_confusion.png',
               report_path='results/reports/05b_cnn_extreme_dropout_report.txt',
               model_name='CNN Dropout Agresivo 0.9 (Underfitting)')
