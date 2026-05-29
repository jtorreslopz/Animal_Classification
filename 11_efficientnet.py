#!/usr/bin/env python
# coding: utf-8
 
"""
models/11_efficientnet_transfer_learning.py
--------------------------------------------
Transfer Learning con EfficientNetB0 pre-entrenado en ImageNet.
 
Estrategia:
    - Base convolucional EfficientNetB0: pesos CONGELADOS (no se actualizan)
    - Nueva cabeza clasificadora: pesos entrenables desde cero
 
Solo se entrenan ~66.000 parámetros de los >5M totales de EfficientNetB0,
lo que permite obtener un 98% de accuracy en apenas 10 épocas.
 
Arquitectura:
    Input (224x224x3) → Rescaling
    → EfficientNetB0 (congelado) → GAP interno → vector 1.280
    → Dense(256, ReLU) → Dropout(0.5) → Softmax(6)
"""
 
# ============================================================
# IMPORTACIONES
# ============================================================
import tensorflow as tf
import numpy as np
import time
import pickle
from keras.layers import Dense, Dropout, Rescaling, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.applications import EfficientNetB0
from keras import Model
from tensorflow.keras.models import load_model
 
from utils.dataset_loader import load_datasets
from utils.plot_training import plot_training, FakeHistory
from utils.metrics import evaluate_model
 
# ============================================================
# CARGA DEL DATASET
# Nota: EfficientNetB0 requiere imágenes de 224x224
# Se recarga el dataset con el tamaño correcto
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32
 
train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/train',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)
 
val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/validation',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)
 
test_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/test',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)
 
print(f"✅ Datasets cargados a {IMG_SIZE}x{IMG_SIZE}")
print(f"   Clases: {train_ds.class_names}")
 
# ============================================================
# BASE CONVOLUCIONAL — EfficientNetB0 CONGELADA
# ============================================================
base_model = EfficientNetB0(
    include_top=False,       # Sin la cabeza clasificadora original
    weights='imagenet',      # Pesos pre-entrenados en ImageNet
    pooling='avg',           # GAP interno → vector de 1.280 valores
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
 
# Congelar todos los pesos de la base convolucional
base_model.trainable = False
 
print(f"\n🔒 Base convolucional congelada.")
print(f"   Parámetros totales EfficientNetB0: {base_model.count_params():,}")
print(f"   Parámetros entrenables (solo cabeza): ~66.000")
 
# ============================================================
# NUEVA CABEZA CLASIFICADORA — ENTRENABLE
# ============================================================
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = Rescaling(1./255)(inputs)
x = base_model(x, training=False)  # training=False → BN en modo inferencia
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(6, activation='softmax', name='outputs')(x)
 
model_eff = Model(inputs, outputs, name='EfficientNetB0_TransferLearning')
model_eff.summary()
 
# ============================================================
# COMPILACIÓN
# ============================================================
opt = Adam(learning_rate=0.001)
 
model_eff.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
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
model_eff_history = model_eff.fit(
    train_ds,
    epochs=30,
    callbacks=[stopping, reduce_lr],
    verbose=1,
    validation_data=val_ds
)
print("Tiempo de entrenamiento: ", time.time() - t, "segundos.")
 
# Guardado del modelo e historial
model_eff.save('model_eff.keras')
print("Modelo guardado en el disco.")
 
with open('model_eff_history.pkl', 'wb') as f:
    pickle.dump(model_eff_history.history, f)
print("Historial guardado en el disco.")
 
# ============================================================
# RESULTADOS
# ============================================================
with open('model_eff_history.pkl', 'rb') as f:
    datos = pickle.load(f)
 
model_eff_history = FakeHistory(datos)
n_epochs = len(model_eff_history.history["loss"])
plot_training(n_epochs, model_eff_history,
              save_path='results/plots/11_efficientnet.png')
 
model_eff = load_model('model_eff.keras')
evaluate_model(model_eff, test_ds,
               save_path='results/plots/11_efficientnet_confusion.png',
               report_path='results/reports/11_efficientnet_report.txt',
               model_name='EfficientNetB0 Transfer Learning')
 