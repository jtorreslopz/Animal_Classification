#!/usr/bin/env python
# coding: utf-8

"""
models/05a_cnn_no_dropout.py
-----------------------------
Experimento de control: CNN básica SIN ninguna capa de Dropout.
Demuestra el overfitting masivo que ocurre en ausencia de regularización.

Arquitectura: igual que CNN básico pero sin Dropout en ninguna capa.
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
# DEFINICIÓN DEL MODELO — Sin Dropout
# ============================================================
cnn_model_nodrop = Sequential()

# Extractor de características
cnn_model_nodrop.add(Input(shape=(299, 299, 3)))
cnn_model_nodrop.add(Rescaling(1./255))

# Bloque 1
cnn_model_nodrop.add(Conv2D(16, 3, padding='same', activation='relu'))
cnn_model_nodrop.add(MaxPooling2D())

# Bloque 2
cnn_model_nodrop.add(Conv2D(32, 3, padding='same', activation='relu'))
cnn_model_nodrop.add(MaxPooling2D())

# Bloque 3
cnn_model_nodrop.add(Conv2D(64, 3, padding='same', activation='relu'))
cnn_model_nodrop.add(MaxPooling2D())
# ❌ Sin Dropout

# Clasificador — Sin Dropout
cnn_model_nodrop.add(Flatten())
cnn_model_nodrop.add(Dense(64, activation='relu'))
# ❌ Sin Dropout
cnn_model_nodrop.add(Dense(6, name="outputs", activation="softmax"))

cnn_model_nodrop.summary()

# ============================================================
# COMPILACIÓN
# ============================================================
opt = Adam(learning_rate=0.001)

cnn_model_nodrop.compile(
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
cnn_history_nodrop = cnn_model_nodrop.fit(
    train_ds,
    epochs=100,
    callbacks=[stopping, reduce_lr],
    verbose=1,
    validation_data=val_ds
)
print("Tiempo de entrenamiento: ", time.time() - t, "segundos.")

# Guardado del modelo e historial
cnn_model_nodrop.save('cnn_model_nodrop.keras')
print("Modelo guardado en el disco.")

with open('cnn_history_nodrop.pkl', 'wb') as f:
    pickle.dump(cnn_history_nodrop.history, f)
print("Historial guardado en el disco.")

# ============================================================
# RESULTADOS
# ============================================================
with open('cnn_history_nodrop.pkl', 'rb') as f:
    datos = pickle.load(f)

cnn_history_nodrop = FakeHistory(datos)
n_epochs = len(cnn_history_nodrop.history["loss"])
plot_training(n_epochs, cnn_history_nodrop,
              save_path='results/plots/05a_cnn_no_dropout.png')

# Cargo el modelo y evalúo
cnn_model_nodrop = load_model('cnn_model_nodrop.keras')
evaluate_model(cnn_model_nodrop, test_ds,
               save_path='results/plots/05a_cnn_no_dropout_confusion.png',
               report_path='results/reports/05a_cnn_no_dropout_report.txt',
               model_name='CNN Sin Dropout (Overfitting)')
