"""
Test to verify the teacher model architecture is correct
"""
import torch
from torchvision import models
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def test():
    device = 'cpu'
    
    # Load the checkpoint
    ckpt_path = 'outputs/models/teacher_resnet50.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Create model correctly
    model = models.resnet50(weights=None)  # Random init
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes
    model = model.to(device)
    
    # Load checkpoint
    model.load_state_dict(ckpt, strict=True)  # strict=True to ensure exact match
    model.eval()
    
    # Get the FC layer weights
    fc_weight = model.fc.weight.data
    fc_bias = model.fc.bias.data
    
    print(f"FC weight shape: {fc_weight.shape}")
    print(f"FC weight mean: {fc_weight.mean():.6f}, std: {fc_weight.std():.6f}")
    print(f"FC bias mean: {fc_bias.mean():.6f}, std: {fc_bias.std():.6f}")
    
    # Check conv1 weights (should have ImageNet values)
    conv1_weight = model.conv1.weight.data
    print(f"\nConv1 weight shape: {conv1_weight.shape}")
    print(f"Conv1 weight mean: {conv1_weight.mean():.6f}, std: {conv1_weight.std():.6f}")
    
    # Test with 100 random inputs
    print("\n=== Testing inference ===")
    model.eval()
    with torch.no_grad():
        outputs = []
        for _ in range(10):
            x = torch.randn(1, 3, 224, 224).to(device)
            out = model(x)
            outputs.append(out.argmax(1).item())
        
        pred_dist = {}
        for p in outputs:
            pred_dist[p] = pred_dist.get(p, 0) + 1
        
        print(f"Predictions over 10 random inputs: {pred_dist}")
        
        # If model is untrained/random, should see roughly uniform distribution
        # If trained, might see some bias
        uniform_dist = sum(1 for count in pred_dist.values() if count == 1)
        print(f"Uniform class distribution (random): {uniform_dist}/10 classes")

if __name__ == '__main__':
    test()
