import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.utils.prune as prune

def train_pruning(model, train_loader, val_loader, device,
                  epochs=10, lr=1e-5, weight_decay=1e-4, prune_amount=0.3, save_path=None):
    """
    Apply pruning to Conv2d and Linear layers, then fine-tune.
    
    Args:
        model: PyTorch model
        train_loader, val_loader: DataLoaders
        device: 'cuda' or 'cpu'
        epochs: Number of fine-tuning epochs
        lr: Learning rate
        weight_decay: Weight decay
        prune_amount: Fraction of weights to prune
        save_path: Path to save best model
    Returns:
        pruned & fine-tuned model, training history
    """
    model.to(device)
    
    # Step 1: Apply pruning
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=prune_amount)
            prune.remove(module, 'weight')  # make pruning permanent

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Pruning Fine-tune"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
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
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
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
            torch.save(model.state_dict(), save_path)
            print(f"Saved best pruned model with accuracy {val_acc:.4f}")
            best_acc = val_acc

    return model, history
