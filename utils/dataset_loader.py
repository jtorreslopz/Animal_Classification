"""
utils/dataset_loader.py
-----------------------
Carga centralizada de los datasets de entrenamiento, validación y test.
Usa las mismas rutas que en los notebooks originales: 'dataset/train', etc.
"""
 
import tensorflow as tf
 
# ============================================================
# CONFIGURACIÓN — mismas rutas y parámetros que en los notebooks
# ============================================================
IMG_WIDTH  = 299
IMG_HEIGHT = 299
BATCH_SIZE = 32
 
TRAIN_PATH = 'dataset/train'
VAL_PATH   = 'dataset/validation'
TEST_PATH  = 'dataset/test'
# ============================================================
 
 
def load_datasets():
    """
    Carga y devuelve los datasets de entrenamiento, validación y test.
 
    Retorna:
    --------
    train_ds, val_ds, test_ds : tf.data.Dataset
 
    Ejemplo de uso:
    ---------------
        from utils.dataset_loader import load_datasets
        train_ds, val_ds, test_ds = load_datasets()
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
 
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_PATH,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
 
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
 
    print(f"✅ Datasets cargados correctamente.")
    print(f"   Clases detectadas: {train_ds.class_names}")
 
    return train_ds, val_ds, test_ds
 