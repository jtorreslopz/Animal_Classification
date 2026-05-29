"""
utils/
------
Módulo de utilidades compartidas para todos los modelos CNN.

Módulos disponibles:
    - plot_training   : visualización de curvas de entrenamiento
    - dataset_loader  : carga centralizada de datasets
    - metrics         : evaluación y matriz de confusión
"""

from utils.plot_training import plot_training, FakeHistory
from utils.dataset_loader import load_datasets
from utils.metrics import evaluate_model
