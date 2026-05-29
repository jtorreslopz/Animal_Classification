"""
utils/metrics.py
----------------
Funciones para evaluar y visualizar los resultados de los modelos CNN:
matriz de confusión e informe de clasificación.
"""
 
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
 
 
def evaluate_model(model, test_ds, save_path=None, report_path=None, model_name="Modelo"):
    """
    Evalúa el modelo sobre el conjunto de test y muestra:
    - Loss y Accuracy global
    - Matriz de confusión visual
    - Informe de clasificación (Precision, Recall, F1-score)
 
    Parámetros:
    -----------
    model : keras.Model
        Modelo entrenado o cargado.
    test_ds : tf.data.Dataset
        Dataset de test.
    save_path : str, opcional
        Guarda la matriz de confusión en esa ruta.
        Ej: 'results/plots/02_cnn_basic_confusion.png'
    report_path : str, opcional
        Guarda el informe de clasificación en esa ruta.
        Ej: 'results/reports/02_cnn_basic_report.txt'
    model_name : str
        Nombre del modelo para títulos y prints.
 
    Retorna:
    --------
    y_true, y_pred : listas con etiquetas reales y predichas.
 
    Ejemplo de uso:
    ---------------
        from utils.metrics import evaluate_model
        y_true, y_pred = evaluate_model(
            model, test_ds,
            save_path='results/plots/02_cnn_basic_confusion.png',
            report_path='results/reports/02_cnn_basic_report.txt',
            model_name='CNN Básico'
        )
    """
 
    # Loss y Accuracy
    score = model.evaluate(test_ds, verbose=0)
    print("-" * 10)
    print(f"Modelo: {model_name}")
    print("Loss:     {:.2f}".format(score[0]))
    print("Accuracy: {:.2f} %".format(score[1] * 100))
 
    # Predicciones
    y_true = []
    y_pred = []
 
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
 
    class_names = test_ds.class_names
 
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicción del modelo')
    plt.ylabel('Etiqueta real')
    plt.title(f'Matriz de Confusión: {model_name}')
    plt.tight_layout()
 
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Matriz guardada en: {save_path}")
 
    plt.show()
 
    # Informe de clasificación
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"\n--- INFORME DE CLASIFICACIÓN: {model_name} ---")
    print(report)
 
    # Guardado del informe en .txt
    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(f"INFORME DE CLASIFICACIÓN: {model_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Loss:     {score[0]:.2f}\n")
            f.write(f"Accuracy: {score[1]*100:.2f} %\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
        print(f"Informe guardado en: {report_path}")
 
    return y_true, y_pred
 

    
