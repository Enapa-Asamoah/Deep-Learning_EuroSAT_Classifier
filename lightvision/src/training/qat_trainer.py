import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.quantization import get_default_qat_qconfig, prepare_qat, convert, fuse_modules

def train_qat(model, train_loader, val_loader, device,
              epochs=10, lr=1e-5, weight_decay=1e-4, save_path=None):
    """
    Quantization-Aware Training (QAT) pipeline for student model.
    """
    model.to(device)
    model.train()

    # Step 1: Fuse layers (Conv + BN + ReLU)
    if isinstance(model, nn.Module):
        # Example: for ResNet
        for module_name, module in model.named_children():
            if module_name == "conv1":
                model = fuse_modules(model, ['conv1', 'bn1', 'relu'], inplace=True)

    # Step 2: Set QAT config
    model.qconfig = get_default_qat_qconfig('fbgemm')
    model_prepared = prepare_qat(model, inplace=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_prepared.parameters(), lr=lr, weight_decay=weight_decay)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    for epoch in range(epochs):
        model_prepared.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - QAT Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_prepared(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model_prepared.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model_prepared(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if save_path and val_acc > best_acc:
            # Convert to quantized model and save
            model_quantized = convert(model_prepared.eval(), inplace=False)
            torch.save(model_quantized.state_dict(), save_path)
            print(f"Saved best QAT model with accuracy {val_acc:.4f}")
            best_acc = val_acc

    return model_quantized, history
