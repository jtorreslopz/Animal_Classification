#!/usr/bin/env python
# coding: utf-8
 
"""
models/02_cnn_basic.py
-----------------------
Red Neuronal Convolucional básica con 3 bloques convolucionales y Flatten.
Modelo CNN de referencia sobre el que se aplican todas las modificaciones posteriores.
 
Arquitectura:
    Input (299x299x3) → Rescaling
    → [Conv2D(16) → MaxPool]
    → [Conv2D(32) → MaxPool]
    → [Conv2D(64) → MaxPool] → Dropout(0.2)
    → Flatten → Dense(128, ReLU) → Dropout(0.3)
    → Dense(6, Softmax)
"""
 
# ============================================================
# IMPORTACIONES
# ============================================================
import numpy as np
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
# DEFINICIÓN DEL MODELO
# ============================================================
cnn_model = Sequential()
 
# Extractor de características
cnn_model.add(Input(shape=(299, 299, 3)))
cnn_model.add(Rescaling(1./255))
 
# Bloque 1: Bordes y colores
cnn_model.add(Conv2D(16, 3, padding='same', activation='relu'))
cnn_model.add(MaxPooling2D())
 
# Bloque 2: Texturas
cnn_model.add(Conv2D(32, 3, padding='same', activation='relu'))
cnn_model.add(MaxPooling2D())
 
# Bloque 3: Formas complejas
cnn_model.add(Conv2D(64, 3, padding='same', activation='relu'))
cnn_model.add(MaxPooling2D())
cnn_model.add(Dropout(0.2))
 
# Clasificador
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation="relu"))
cnn_model.add(Dropout(0.3))
cnn_model.add(Dense(6, name="outputs", activation="softmax"))
 
cnn_model.summary()
 
# ============================================================
# COMPILACIÓN
# ============================================================
opt = Adam(learning_rate=0.001)
 
cnn_model.compile(
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
cnn_history = cnn_model.fit(
    train_ds,
    epochs=100,
    callbacks=[stopping, reduce_lr],
    verbose=1,
    validation_data=val_ds
)
print("Tiempo de entrenamiento: ", time.time() - t, "segundos.")
 
# Guardado del modelo e historial
cnn_model.save('modelo_cnn.keras')
print("Modelo guardado en el disco.")
 
with open('historial_cnn.pkl', 'wb') as f:
    pickle.dump(cnn_history.history, f)
print("Historial guardado en el disco.")
 
# ============================================================
# RESULTADOS
# ============================================================
with open('historial_cnn.pkl', 'rb') as f:
    datos = pickle.load(f)
 
cnn_history = FakeHistory(datos)
n_epochs = len(cnn_history.history["loss"])
plot_training(n_epochs, cnn_history, save_path='results/plots/02_cnn_basic.png')
 
# Cargo el modelo y evalúo
cnn_model = load_model('modelo_cnn.keras')
evaluate_model(cnn_model, test_ds,
               save_path='results/plots/02_cnn_basic_confusion.png',
               report_path='results/reports/02_cnn_basic_report.txt',
               model_name='CNN Básico (3 Bloques + Flatten)')