"""
utils/plot_training.py
----------------------
Funciones para visualizar las curvas de entrenamiento de los modelos CNN.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_training(n_epochs, mfit, save_path=None):
    """
    Genera las curvas de Accuracy y Loss para entrenamiento y validación.

    Parámetros:
    -----------
    n_epochs : int
        Número de épocas entrenadas (len del historial).
    mfit : objeto con atributo .history (dict)
        Historial de entrenamiento. Puede ser el objeto devuelto por model.fit()
        o un FakeHistory cargado desde pickle.
    save_path : str, opcional
        Si se especifica, guarda la figura en esa ruta (ej: 'results/plots/modelo_gap.png').
        Si es None, simplemente muestra la figura.

    Ejemplo de uso:
    ---------------
        from utils.plot_training import plot_training
        n_epochs = len(history.history["loss"])
        plot_training(n_epochs, history)
    """
    N = n_epochs
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Training Loss and Accuracy')

    # Accuracy
    ax1.plot(np.arange(0, N), mfit.history["accuracy"], label="train")
    ax1.plot(np.arange(0, N), mfit.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch #")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")

    # Loss
    ax2.plot(np.arange(0, N), mfit.history["loss"], label="train")
    ax2.plot(np.arange(0, N), mfit.history["val_loss"], label="val")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch #")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfica guardada en: {save_path}")

    plt.show()


class FakeHistory:
    """
    Clase auxiliar para reconstruir un objeto history a partir
    de un diccionario cargado con pickle.

    Ejemplo de uso:
    ---------------
        import pickle
        with open('cnn_history_gap.pkl', 'rb') as f:
            datos = pickle.load(f)
        history = FakeHistory(datos)
        plot_training(len(history.history["loss"]), history)
    """
    def __init__(self, history_dict):
        self.history = history_dict
