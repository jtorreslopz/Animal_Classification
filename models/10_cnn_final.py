#!/usr/bin/env python
# coding: utf-8

"""
models/10_cnn_final.py
-----------------------
Modelo final de desarrollo propio: combina todas las técnicas que han
demostrado ser eficaces a lo largo del estudio.

Técnicas combinadas:
    - Data Augmentation (flip + rotación)
    - Batch Normalization (Conv → BN → ReLU → MaxPool)
    - Global Average Pooling (en lugar de Flatten)
    - 4 bloques convolucionales profundos (32/64/128/256)

Arquitectura:
    DA → 4 Bloques [Conv → BN → ReLU → MaxPool] → Dropout(0.2)
    → GAP → Dense(64) → BN → ReLU → Dropout(0.2) → Softmax(6)
"""

# ============================================================
# IMPORTACIONES
# ============================================================
import keras
import time
import pickle
from keras.layers import (GlobalAveragePooling2D, Input, Dense, Dropout,
                           Conv2D, MaxPooling2D, Rescaling,
                           BatchNormalization, Activation,
                           RandomFlip, RandomRotation)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from utils.dataset_loader import load_datasets
from utils.plot_training import plot_training, FakeHistory
from utils.metrics import evaluate_model

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
])

# ============================================================
# DEFINICIÓN DEL MODELO
# ============================================================
cnn_3_4_model_deepGAP_64 = Sequential()

# Preprocesado + Data Augmentation
cnn_3_4_model_deepGAP_64.add(Input(shape=(299, 299, 3)))
cnn_3_4_model_deepGAP_64.add(Rescaling(1./255))
cnn_3_4_model_deepGAP_64.add(data_augmentation)  # ✅ Bug corregido: DA ahora se aplica

# Bloque 1: Bordes y colores
cnn_3_4_model_deepGAP_64.add(Conv2D(32, 3, padding='same'))
cnn_3_4_model_deepGAP_64.add(BatchNormalization())
cnn_3_4_model_deepGAP_64.add(Activation('relu'))
cnn_3_4_model_deepGAP_64.add(MaxPooling2D())

# Bloque 2: Texturas
cnn_3_4_model_deepGAP_64.add(Conv2D(64, 3, padding='same'))
cnn_3_4_model_deepGAP_64.add(BatchNormalization())
cnn_3_4_model_deepGAP_64.add(Activation('relu'))
cnn_3_4_model_deepGAP_64.add(MaxPooling2D())

# Bloque 3: Formas complejas
cnn_3_4_model_deepGAP_64.add(Conv2D(128, 3, padding='same'))
cnn_3_4_model_deepGAP_64.add(BatchNormalization())
cnn_3_4_model_deepGAP_64.add(Activation('relu'))
cnn_3_4_model_deepGAP_64.add(MaxPooling2D())

# Bloque 4: Rasgos específicos
cnn_3_4_model_deepGAP_64.add(Conv2D(256, 3, padding='same'))
cnn_3_4_model_deepGAP_64.add(BatchNormalization())
cnn_3_4_model_deepGAP_64.add(Activation('relu'))
cnn_3_4_model_deepGAP_64.add(MaxPooling2D())
cnn_3_4_model_deepGAP_64.add(Dropout(0.2))

# Clasificador con GAP
cnn_3_4_model_deepGAP_64.add(GlobalAveragePooling2D())
cnn_3_4_model_deepGAP_64.add(Dense(64))
cnn_3_4_model_deepGAP_64.add(BatchNormalization())
cnn_3_4_model_deepGAP_64.add(Activation('relu'))
cnn_3_4_model_deepGAP_64.add(Dropout(0.2))
cnn_3_4_model_deepGAP_64.add(Dense(6, name="outputs", activation="softmax"))

cnn_3_4_model_deepGAP_64.summary()

# ============================================================
# COMPILACIÓN
# ============================================================
opt = Adam(learning_rate=0.001)

cnn_3_4_model_deepGAP_64.compile(
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
    min_lr=1e-7
)

# ============================================================
# ENTRENAMIENTO
# ============================================================
t = time.time()
cnn_history_3_4_model_deepGAP_64 = cnn_3_4_model_deepGAP_64.fit(
    train_ds,
    epochs=100,
    callbacks=[stopping, reduce_lr],
    verbose=1,
    validation_data=val_ds
)
print("Tiempo de entrenamiento: ", time.time() - t, "segundos.")

# Guardado del modelo e historial
cnn_3_4_model_deepGAP_64.save('cnn_3_4_model_deepGAP_64.keras')
print("Modelo guardado en el disco.")

with open('cnn_history_3_4_model_deepGAP_64.pkl', 'wb') as f:
    pickle.dump(cnn_history_3_4_model_deepGAP_64.history, f)
print("Historial guardado en el disco.")

# ============================================================
# RESULTADOS
# ============================================================
with open('cnn_history_3_4_model_deepGAP_64.pkl', 'rb') as f:
    datos = pickle.load(f)

cnn_history_3_4_model_deepGAP_64 = FakeHistory(datos)
n_epochs = len(cnn_history_3_4_model_deepGAP_64.history["loss"])
plot_training(n_epochs, cnn_history_3_4_model_deepGAP_64,
              save_path='results/plots/10_cnn_final.png')

cnn_3_4_model_deepGAP_64 = load_model('cnn_3_4_model_deepGAP_64.keras')
evaluate_model(cnn_3_4_model_deepGAP_64, test_ds,
               save_path='results/plots/10_cnn_final_confusion.png',
               report_path='results/reports/10_cnn_final_report.txt',
               model_name='Modelo Final (4 Bloques + GAP + BN + DA)')
