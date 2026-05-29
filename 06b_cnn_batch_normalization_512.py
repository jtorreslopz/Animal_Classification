#!/usr/bin/env python
# coding: utf-8

"""
models/06_cnn_batch_normalization.py
--------------------------------------
CNN básica (3 bloques) con Batch Normalization tras cada capa convolucional
y en el clasificador. Estudia el efecto aislado de BN sobre una red simple.

Secuencia en cada bloque: Conv2D → BN → ReLU → MaxPooling
"""

# ============================================================
# IMPORTACIONES
# ============================================================
import time
import pickle
from keras.layers import (Flatten, Input, Dense, Dropout,
                           Conv2D, MaxPooling2D, Rescaling,
                           BatchNormalization, Activation)
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
cnn_model_batch = Sequential()

# Bloque 1: Bordes y colores
cnn_model_batch.add(Input(shape=(299, 299, 3)))
cnn_model_batch.add(Rescaling(1./255))
cnn_model_batch.add(Conv2D(16, 3, padding='same'))
cnn_model_batch.add(BatchNormalization())
cnn_model_batch.add(Activation('relu'))
cnn_model_batch.add(MaxPooling2D())

# Bloque 2: Texturas
cnn_model_batch.add(Conv2D(32, 3, padding='same'))
cnn_model_batch.add(BatchNormalization())
cnn_model_batch.add(Activation('relu'))
cnn_model_batch.add(MaxPooling2D())

# Bloque 3: Formas complejas
cnn_model_batch.add(Conv2D(64, 3, padding='same'))
cnn_model_batch.add(BatchNormalization())
cnn_model_batch.add(Activation('relu'))
cnn_model_batch.add(MaxPooling2D())

# Clasificador
cnn_model_batch.add(Flatten())
cnn_model_batch.add(Dense(512))
cnn_model_batch.add(BatchNormalization())
cnn_model_batch.add(Activation('relu'))
cnn_model_batch.add(Dropout(0.5))
cnn_model_batch.add(Dense(6, name="outputs", activation="softmax"))

cnn_model_batch.summary()

# ============================================================
# COMPILACIÓN
# ============================================================
opt = Adam(learning_rate=0.001)

cnn_model_batch.compile(
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
cnn_batch_history = cnn_model_batch.fit(
    train_ds,
    epochs=100,
    callbacks=[stopping, reduce_lr],
    verbose=1,
    validation_data=val_ds
)
print("Tiempo de entrenamiento: ", time.time() - t, "segundos.")

# Guardado del modelo e historial
cnn_model_batch.save('cnn_model_batch.keras')
print("Modelo guardado en el disco.")

with open('cnn_batch_history.pkl', 'wb') as f:
    pickle.dump(cnn_batch_history.history, f)
print("Historial guardado en el disco.")

# ============================================================
# RESULTADOS
# ============================================================
with open('cnn_batch_history.pkl', 'rb') as f:
    datos = pickle.load(f)

cnn_batch_history = FakeHistory(datos)
n_epochs = len(cnn_batch_history.history["loss"])
plot_training(n_epochs, cnn_batch_history,
              save_path='results/plots/06_cnn_batch_normalization.png')

# Cargo el modelo y evalúo
cnn_model_batch = load_model('cnn_model_batch.keras')
evaluate_model(cnn_model_batch, test_ds,
               save_path='results/plots/06_cnn_batch_normalization_confusion.png',
               report_path='results/reports/06_cnn_batch_normalization_report.txt',
               model_name='CNN + Batch Normalization')
