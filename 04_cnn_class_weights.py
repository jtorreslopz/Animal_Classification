#!/usr/bin/env python
# coding: utf-8

"""
models/04_cnn_class_weights.py
--------------------------------
CNN básica con Class Weights para combatir el desbalanceo de clases.
Los pesos se calculan de forma inversamente proporcional a la frecuencia
de cada clase en el dataset de entrenamiento.

Arquitectura: idéntica al CNN básico. La diferencia está en la función
de pérdida, que penaliza más los errores en clases minoritarias.
"""

# ============================================================
# IMPORTACIONES
# ============================================================
import numpy as np
import time
import pickle
from sklearn.utils import class_weight
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
# CÁLCULO DE CLASS WEIGHTS
# ============================================================
# Obtengo las etiquetas del dataset de entrenamiento
y_train = np.concatenate([y for x, y in train_ds], axis=0)

# Calculo los pesos inversamente proporcionales a la frecuencia de cada clase
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Convierto a diccionario para Keras (índice → peso)
class_weights_dict = dict(enumerate(weights))
print("Pesos calculados:", class_weights_dict)
print("Clases:", train_ds.class_names)

# ============================================================
# DEFINICIÓN DEL MODELO
# ============================================================
cnn_model_classweight = Sequential()

# Extractor de características
cnn_model_classweight.add(Input(shape=(299, 299, 3)))
cnn_model_classweight.add(Rescaling(1./255))

# Bloque 1: Bordes y colores
cnn_model_classweight.add(Conv2D(16, 3, padding='same', activation='relu'))
cnn_model_classweight.add(MaxPooling2D())

# Bloque 2: Texturas
cnn_model_classweight.add(Conv2D(32, 3, padding='same', activation='relu'))
cnn_model_classweight.add(MaxPooling2D())

# Bloque 3: Formas complejas
cnn_model_classweight.add(Conv2D(64, 3, padding='same', activation='relu'))
cnn_model_classweight.add(MaxPooling2D())
cnn_model_classweight.add(Dropout(0.2))

# Clasificador
cnn_model_classweight.add(Flatten())
cnn_model_classweight.add(Dense(128, activation="relu"))
cnn_model_classweight.add(Dropout(0.3))
cnn_model_classweight.add(Dense(6, name="outputs", activation="softmax"))

cnn_model_classweight.summary()

# ============================================================
# COMPILACIÓN
# ============================================================
opt = Adam(learning_rate=0.001)

cnn_model_classweight.compile(
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
cnn_classweight_history = cnn_model_classweight.fit(
    train_ds,
    epochs=100,
    class_weight=class_weights_dict,
    callbacks=[stopping, reduce_lr],
    verbose=1,
    validation_data=val_ds
)
print("Tiempo de entrenamiento: ", time.time() - t, "segundos.")

# Guardado del modelo e historial
cnn_model_classweight.save('modelo_cnn_classweight.keras')
print("Modelo guardado en el disco.")

with open('historial_cnn_classweight.pkl', 'wb') as f:
    pickle.dump(cnn_classweight_history.history, f)
print("Historial guardado en el disco.")

# ============================================================
# RESULTADOS
# ============================================================
with open('historial_cnn_classweight.pkl', 'rb') as f:
    datos = pickle.load(f)

cnn_classweight_history = FakeHistory(datos)
n_epochs = len(cnn_classweight_history.history["loss"])
plot_training(n_epochs, cnn_classweight_history,
              save_path='results/plots/04_cnn_class_weights.png')

# Cargo el modelo y evalúo sobre test_ds (no val_ds)
cnn_model_classweight = load_model('modelo_cnn_classweight.keras')
evaluate_model(cnn_model_classweight, test_ds,
               save_path='results/plots/04_cnn_class_weights_confusion.png',
               report_path='results/reports/04_cnn_class_weights_report.txt',
               model_name='CNN + Class Weights')
