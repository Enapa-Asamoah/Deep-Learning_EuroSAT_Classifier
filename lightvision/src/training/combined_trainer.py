import torch
from tqdm import tqdm

from src.training.student_trainer import train_student
from src.training.distillation_trainer import train_distillation
from src.training.qat_trainer import train_qat
from src.training.pruning_trainer import train_pruning

def train_combined(student_model, teacher_model, train_loader, val_loader, device,
                   steps=['pruning', 'distillation', 'qat'], epochs_dict=None, lr_dict=None, alpha=0.5, temperature=5.0,
                   save_path=None):
    """
    Apply multiple compression techniques sequentially with fine-tuning

    steps: list of strings, any combination of ['pruning', 'distillation', 'qat']
    epochs_dict: dict mapping step -> epochs
    lr_dict: dict mapping step -> learning rate
    Returns final model and combined history dictionary
    """

    if epochs_dict is None:
        epochs_dict = {'pruning': 10, 'distillation': 20, 'qat': 10}
    if lr_dict is None:
        lr_dict = {'pruning': 1e-5, 'distillation': 1e-4, 'qat': 1e-5}

    combined_history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'step': []
    }

    current_model = student_model

    for step in steps:
        print(f"\n=== Applying step: {step} ===")

        if step == 'pruning':
            current_model, history = train_pruning(current_model, train_loader, val_loader, device,
                                                  epochs=epochs_dict.get('pruning', 10),
                                                  lr=lr_dict.get('pruning', 1e-5),
                                                  save_path=save_path)
        elif step == 'distillation':
            if teacher_model is None:
                raise ValueError('Teacher model must be provided for distillation')
            current_model, history = train_distillation(current_model, teacher_model, train_loader, val_loader, device,
                                                        epochs=epochs_dict.get('distillation', 20),
                                                        lr=lr_dict.get('distillation', 1e-4),
                                                        alpha=alpha, temperature=temperature,
                                                        save_path=save_path)
        elif step == 'qat':
            current_model, history = train_qat(current_model, train_loader, val_loader, device,
                                              epochs=epochs_dict.get('qat', 10),
                                              lr=lr_dict.get('qat', 1e-5),
                                              save_path=save_path)
        else:
            raise ValueError(f'Unknown step: {step}')

        # Append step info to history
        for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            combined_history[key].extend(history[key])
        combined_history['step'].extend([step]*len(history['train_loss']))

    return current_model, combined_history
