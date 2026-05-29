#!/usr/bin/env python
# coding: utf-8

"""
models/08_cnn_deep.py
----------------------
CNN con mayor profundidad: 4 bloques convolucionales con progresión
de filtros 32 → 64 → 128 → 256. Sin Batch Normalization.
Estudia el efecto aislado de la profundidad sobre el rendimiento.

Arquitectura:
    4 Bloques Conv (32/64/128/256) → Dropout(0.3)
    → Flatten → Dense(256, ReLU) → Dropout(0.5) → Softmax(6)
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
# DEFINICIÓN DEL MODELO
# ============================================================
cnn_model_deep = Sequential()

# Bloque 1: Bordes y colores
cnn_model_deep.add(Input(shape=(299, 299, 3)))
cnn_model_deep.add(Rescaling(1./255))
cnn_model_deep.add(Conv2D(32, 3, padding='same', activation='relu'))
cnn_model_deep.add(MaxPooling2D())

# Bloque 2: Texturas
cnn_model_deep.add(Conv2D(64, 3, padding='same', activation='relu'))
cnn_model_deep.add(MaxPooling2D())

# Bloque 3: Formas complejas
cnn_model_deep.add(Conv2D(128, 3, padding='same', activation='relu'))
cnn_model_deep.add(MaxPooling2D())

# Bloque 4: Rasgos específicos
cnn_model_deep.add(Conv2D(256, 3, padding='same', activation='relu'))
cnn_model_deep.add(MaxPooling2D())
cnn_model_deep.add(Dropout(0.3))

# Clasificador
cnn_model_deep.add(Flatten())
cnn_model_deep.add(Dense(128, activation='relu'))
cnn_model_deep.add(Dropout(0.3))
cnn_model_deep.add(Dense(6, name="outputs", activation="softmax"))

cnn_model_deep.summary()

# ============================================================
# COMPILACIÓN
# ============================================================
opt = Adam(learning_rate=0.001)

cnn_model_deep.compile(
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
cnn_history_deep = cnn_model_deep.fit(
    train_ds,
    epochs=100,
    callbacks=[stopping, reduce_lr],
    verbose=1,
    validation_data=val_ds
)
print("Tiempo de entrenamiento: ", time.time() - t, "segundos.")

# Guardado del modelo e historial
cnn_model_deep.save('cnn_model_deep.keras')
print("Modelo guardado en el disco.")

with open('cnn_history_deep.pkl', 'wb') as f:
    pickle.dump(cnn_history_deep.history, f)
print("Historial guardado en el disco.")

# ============================================================
# RESULTADOS
# ============================================================
with open('cnn_history_deep.pkl', 'rb') as f:
    datos = pickle.load(f)

cnn_history_deep = FakeHistory(datos)
n_epochs = len(cnn_history_deep.history["loss"])
plot_training(n_epochs, cnn_history_deep,
              save_path='results/plots/08_cnn_deep.png')

# Carga de modelo profundo
cnn_model_deep = load_model('cnn_model_deep.keras')
evaluate_model(cnn_model_deep, test_ds,
               save_path='results/plots/08_cnn_deep_confusion.png',
               report_path='results/reports/08_cnn_deep_report.txt',
               model_name='CNN Mayor Profundidad (4 Bloques + Flatten)')
