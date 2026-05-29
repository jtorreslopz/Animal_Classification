#!/usr/bin/env python
# coding: utf-8

"""
models/07_cnn_gap.py
---------------------
CNN básica (3 bloques) con Global Average Pooling en lugar de Flatten.
GAP reduce cada mapa de características a un único valor (su promedio),
eliminando millones de parámetros y actuando como regularizador estructural.

Arquitectura:
    3 Bloques Conv → GAP → Dense(64, ReLU) → Dropout(0.2) → Softmax(6)
"""

# ============================================================
# IMPORTACIONES
# ============================================================
import time
import pickle
from keras.layers import (GlobalAveragePooling2D, Input, Dense, Dropout,
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
cnn_model_GAP = Sequential()

# Extractor de características
cnn_model_GAP.add(Input(shape=(299, 299, 3)))
cnn_model_GAP.add(Rescaling(1./255))

# Bloque 1: Bordes y colores
cnn_model_GAP.add(Conv2D(16, 3, padding='same', activation='relu'))
cnn_model_GAP.add(MaxPooling2D())

# Bloque 2: Texturas
cnn_model_GAP.add(Conv2D(32, 3, padding='same', activation='relu'))
cnn_model_GAP.add(MaxPooling2D())

# Bloque 3: Formas complejas
cnn_model_GAP.add(Conv2D(64, 3, padding='same', activation='relu'))
cnn_model_GAP.add(MaxPooling2D())
cnn_model_GAP.add(Dropout(0.3))

# Clasificador con GAP en lugar de Flatten
cnn_model_GAP.add(GlobalAveragePooling2D())  # 64 mapas → vector de 64 valores
cnn_model_GAP.add(Dense(64, activation='relu'))
cnn_model_GAP.add(Dropout(0.2))
cnn_model_GAP.add(Dense(6, name="outputs", activation="softmax"))

cnn_model_GAP.summary()

# ============================================================
# COMPILACIÓN
# ============================================================
opt = Adam(learning_rate=0.001)

cnn_model_GAP.compile(
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
cnn_history_GAP = cnn_model_GAP.fit(
    train_ds,
    epochs=100,
    callbacks=[stopping, reduce_lr],
    verbose=1,
    validation_data=val_ds
)
print("Tiempo de entrenamiento: ", time.time() - t, "segundos.")

# Guardado del modelo e historial
cnn_model_GAP.save('cnn_model_GAP.keras')
print("Modelo guardado en el disco.")

with open('cnn_history_GAP.pkl', 'wb') as f:
    pickle.dump(cnn_history_GAP.history, f)
print("Historial guardado en el disco.")

# ============================================================
# RESULTADOS
# ============================================================
with open('cnn_history_GAP.pkl', 'rb') as f:
    datos = pickle.load(f)

cnn_history_GAP = FakeHistory(datos)
n_epochs = len(cnn_history_GAP.history["loss"])
plot_training(n_epochs, cnn_history_GAP,
              save_path='results/plots/07_cnn_gap.png')

# Cargo el modelo y evalúo
cnn_model_GAP = load_model('cnn_model_GAP.keras')
evaluate_model(cnn_model_GAP, test_ds,
               save_path='results/plots/07_cnn_gap_confusion.png',
               report_path='results/reports/07_cnn_gap_report.txt',
               model_name='CNN + Global Average Pooling (64 neuronas)')
