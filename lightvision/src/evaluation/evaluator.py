import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaluate_model(model, data_loader, device, classes, history=None, output_dir="outputs/plots", model_name="model"):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for test data
        device: torch device
        classes: list of class names
        history: optional training history dict
        output_dir: directory to save plots
        model_name: name for saving plots
        
    Returns:
        metrics: dict with accuracy, precision, recall, f1
        y_true: ground truth labels
        y_pred: predicted labels
        plot_paths: dict with paths to saved plots
    """
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'classification_report': classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    }
    
    # Save confusion matrix
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    plot_paths = {'confusion_matrix': cm_path}
    
    print(f"\nEvaluation Results for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nConfusion matrix saved to: {cm_path}")
    
    return metrics, y_true, y_pred, plot_paths
