#!/usr/bin/env python
# coding: utf-8

"""
models/09_cnn_deep_bn.py
--------------------------
CNN profunda (4 bloques) con Batch Normalization tras cada capa convolucional
y en el clasificador. Combinación que resuelve el desbalanceo de clases.

Secuencia en cada bloque: Conv2D → BN → ReLU → MaxPooling

Arquitectura:
    4 Bloques [Conv(32/64/128/256) → BN → ReLU → MaxPool] → Dropout(0.3)
    → Flatten → Dense(128) → BN → ReLU → Dropout(0.5) → Softmax(6)
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
cnn_model_deep_batch = Sequential()

# Bloque 1: Bordes y colores
cnn_model_deep_batch.add(Input(shape=(299, 299, 3)))
cnn_model_deep_batch.add(Rescaling(1./255))
cnn_model_deep_batch.add(Conv2D(32, 3, padding='same'))
cnn_model_deep_batch.add(BatchNormalization())
cnn_model_deep_batch.add(Activation('relu'))
cnn_model_deep_batch.add(MaxPooling2D())

# Bloque 2: Texturas
cnn_model_deep_batch.add(Conv2D(64, 3, padding='same'))
cnn_model_deep_batch.add(BatchNormalization())
cnn_model_deep_batch.add(Activation('relu'))
cnn_model_deep_batch.add(MaxPooling2D())

# Bloque 3: Formas complejas
cnn_model_deep_batch.add(Conv2D(128, 3, padding='same'))
cnn_model_deep_batch.add(BatchNormalization())
cnn_model_deep_batch.add(Activation('relu'))
cnn_model_deep_batch.add(MaxPooling2D())

# Bloque 4: Rasgos específicos
cnn_model_deep_batch.add(Conv2D(256, 3, padding='same'))
cnn_model_deep_batch.add(BatchNormalization())
cnn_model_deep_batch.add(Activation('relu'))
cnn_model_deep_batch.add(MaxPooling2D())
cnn_model_deep_batch.add(Dropout(0.3))

# Clasificador
cnn_model_deep_batch.add(Flatten())
cnn_model_deep_batch.add(Dense(128))
cnn_model_deep_batch.add(BatchNormalization())
cnn_model_deep_batch.add(Activation('relu'))
cnn_model_deep_batch.add(Dropout(0.3))
cnn_model_deep_batch.add(Dense(6, name="outputs", activation="softmax"))

cnn_model_deep_batch.summary()

# ============================================================
# COMPILACIÓN
# ============================================================
opt = Adam(learning_rate=0.001)

cnn_model_deep_batch.compile(
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
cnn_history_deep_batch = cnn_model_deep_batch.fit(
    train_ds,
    epochs=100,
    callbacks=[stopping, reduce_lr],
    verbose=1,
    validation_data=val_ds
)
print("Tiempo de entrenamiento: ", time.time() - t, "segundos.")

# Guardado del modelo e historial
cnn_model_deep_batch.save('cnn_model_deep_batch.keras')
print("Modelo guardado en el disco.")

with open('cnn_history_deep_batch.pkl', 'wb') as f:
    pickle.dump(cnn_history_deep_batch.history, f)
print("Historial guardado en el disco.")

# ============================================================
# RESULTADOS
# ============================================================
with open('cnn_history_deep_batch.pkl', 'rb') as f:
    datos = pickle.load(f)

cnn_history_deep_batch = FakeHistory(datos)
n_epochs = len(cnn_history_deep_batch.history["loss"])
plot_training(n_epochs, cnn_history_deep_batch,
              save_path='results/plots/09_cnn_deep_bn.png')

# ✅ Bug corregido: original iteraba sobre val_ds en lugar de test_ds
cnn_model_deep_batch = load_model('cnn_model_deep_batch.keras')
evaluate_model(cnn_model_deep_batch, test_ds,
               save_path='results/plots/09_cnn_deep_bn_confusion.png',
               report_path='results/reports/09_cnn_deep_bn_report.txt',
               model_name='CNN Profunda + Batch Normalization')
