import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_distillation(student_model, teacher_model, train_loader, val_loader, device,
                       epochs=20, lr=1e-4, alpha=0.5, temperature=5.0, weight_decay=1e-4, save_path=None):
    """
    Train student model with knowledge distillation from teacher
    Args:
        student_model: Student network (nn.Module)
        teacher_model: Pretrained teacher network (nn.Module)
        train_loader, val_loader: PyTorch DataLoaders
        device: 'cuda' or 'cpu'
        epochs: Number of epochs
        lr: Learning rate
        alpha: Weight for soft loss
        temperature: Temperature for distillation
        weight_decay: Weight decay
        save_path: Path to save best student model
    Returns:
        trained student model and history dictionary
    """

    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    teacher_model.eval()  # Freeze teacher

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training" ):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward
            student_outputs = student_model(images)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)

            # Compute distillation loss
            soft_targets = nn.functional.log_softmax(student_outputs/temperature, dim=1)
            teacher_soft = nn.functional.softmax(teacher_outputs/temperature, dim=1)
            loss_kd = criterion_kd(soft_targets, teacher_soft) * (temperature**2)
            loss_ce = criterion_ce(student_outputs, labels)
            loss = alpha * loss_kd + (1 - alpha) * loss_ce

            # Backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = student_outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        student_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student_model(images)
                loss = criterion_ce(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        # Save epoch history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if save_path and val_acc > best_acc:
            torch.save(student_model.state_dict(), save_path)
            print(f"Saved best student model with accuracy {val_acc:.4f}")
            best_acc = val_acc

    return student_model, history