"""
Minimal test to verify checkpoint loading works correctly
"""
import torch
from torchvision import models
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

device = 'cpu'

def test_checkpoint_load():
    """Test loading teacher checkpoint"""
    
    # Create model
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    
    # Load checkpoint
    ckpt_path = 'outputs/models/teacher_resnet50.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    
    print(f"Checkpoint type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"Checkpoint keys: {list(ckpt.keys())[:10]}")
        
        # Check if it's a wrapped checkpoint
        for key in ["model_state_dict", "state_dict"]:
            if key in ckpt:
                print(f"Found wrapper key: {key}")
                ckpt = ckpt[key]
                break
    
    # Try loading
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"Missing keys: {len(missing)} - {list(missing)[:3] if missing else 'none'}")
    print(f"Unexpected keys: {len(unexpected)} - {list(unexpected)[:3] if unexpected else 'none'}")
    
    # Test inference
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy)
    
    print(f"Output shape: {output.shape}")
    pred = output.argmax(1).item()
    print(f"Predicted class: {pred} (0-9 is valid for EuroSAT)")
    
    # Check weights
    first_conv_weight = model.conv1.weight.data
    print(f"\nFirst conv weight - mean: {first_conv_weight.mean():.6f}, std: {first_conv_weight.std():.6f}")
    
    # Compare to untrained
    untrained = models.resnet50(weights=None)
    untrained.fc = torch.nn.Linear(untrained.fc.in_features, 10)
    untrained_weight = untrained.conv1.weight.data
    print(f"Untrained conv weight - mean: {untrained_weight.mean():.6f}, std: {untrained_weight.std():.6f}")
    
    if abs(first_conv_weight.std().item() - untrained_weight.std().item()) > 0.01:
        print("✓ Checkpoint weights appear trained (different from random init)")
    else:
        print("✗ Checkpoint weights look like random init")

if __name__ == '__main__':
    test_checkpoint_load()
