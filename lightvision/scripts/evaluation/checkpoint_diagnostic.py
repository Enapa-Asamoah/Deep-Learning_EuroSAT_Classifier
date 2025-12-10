"""
Diagnostic script to verify checkpoint weights are properly trained (not random)
"""
import torch
import os
import json
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torchvision import models
import numpy as np

CHECKPOINT_DIR = 'outputs/models/'
STATS_OUTPUT = 'outputs/reports/checkpoint_stats.json'

def analyze_layer_stats(model, layer_name):
    """Extract weight statistics from a specific layer"""
    layer = dict(model.named_parameters())[layer_name]
    w = layer.data.cpu().numpy()
    
    return {
        'mean': float(np.mean(w)),
        'std': float(np.std(w)),
        'min': float(np.min(w)),
        'max': float(np.max(w)),
        'shape': list(w.shape),
    }

def get_first_conv_layer_stats(model):
    """Get stats from the first conv layer"""
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) >= 3:  # Conv layer
            w = param.data.cpu().numpy()
            return {
                'layer': name,
                'mean': float(np.mean(w)),
                'std': float(np.std(w)),
                'min': float(np.min(w)),
                'max': float(np.max(w)),
            }
    return None

def get_random_resnet_stats(architecture='resnet50'):
    """Get stats from a freshly initialized ResNet (random baseline)"""
    if architecture == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)  # 10 classes for EuroSAT
    else:
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)  # 10 classes for EuroSAT
    
    return get_first_conv_layer_stats(model)

def main():
    os.makedirs(os.path.dirname(STATS_OUTPUT), exist_ok=True)
    results = {}
    
    # Get random baseline stats
    print("Getting random ResNet50 baseline...")
    resnet50_random = get_random_resnet_stats('resnet50')
    print(f"Random ResNet50 first conv: mean={resnet50_random['mean']:.6f}, std={resnet50_random['std']:.6f}")
    
    print("Getting random ResNet18 baseline...")
    resnet18_random = get_random_resnet_stats('resnet18')
    print(f"Random ResNet18 first conv: mean={resnet18_random['mean']:.6f}, std={resnet18_random['std']:.6f}")
    
    # Analyze saved checkpoints
    for filename in os.listdir(CHECKPOINT_DIR):
        if not filename.endswith('.pth'):
            continue
            
        filepath = os.path.join(CHECKPOINT_DIR, filename)
        print(f"\nAnalyzing {filename}...")
        
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Handle both direct state_dict and wrapped checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Determine architecture
            if 'teacher' in filename or 'pruned' in filename:
                model = models.resnet50(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, 10)  # 10 classes
                arch = 'resnet50'
                baseline = resnet50_random
            else:
                model = models.resnet18(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, 10)  # 10 classes
                arch = 'resnet18'
                baseline = resnet18_random
            
            model.load_state_dict(state_dict, strict=False)
            stats = get_first_conv_layer_stats(model)
            
            # Compare to random baseline
            if stats:
                mean_diff = abs(stats['mean'] - baseline['mean'])
                std_diff = abs(stats['std'] - baseline['std'])
                
                results[filename] = {
                    'architecture': arch,
                    'layer': stats['layer'],
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'random_baseline_mean': baseline['mean'],
                    'random_baseline_std': baseline['std'],
                    'mean_diff_from_random': float(mean_diff),
                    'std_diff_from_random': float(std_diff),
                    'likely_trained': mean_diff > 0.001 and std_diff > 0.001,
                }
                
                print(f"  Layer: {stats['layer']}")
                print(f"  Mean: {stats['mean']:.6f} (random: {baseline['mean']:.6f}, diff: {mean_diff:.6f})")
                print(f"  Std:  {stats['std']:.6f} (random: {baseline['std']:.6f}, diff: {std_diff:.6f})")
                print(f"  ✓ Likely trained" if results[filename]['likely_trained'] else "  ✗ Looks like random init!")
        
        except Exception as e:
            print(f"  ✗ Error loading: {e}")
            results[filename] = {'error': str(e)}
    
    # Save results
    with open(STATS_OUTPUT, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved analysis to {STATS_OUTPUT}")
    
    # Summary
    print("\n=== SUMMARY ===")
    trained_count = sum(1 for v in results.values() if v.get('likely_trained', False))
    untrained_count = sum(1 for v in results.values() if v.get('likely_trained', False) == False)
    print(f"Likely trained: {trained_count}")
    print(f"Likely random: {untrained_count}")

if __name__ == '__main__':
    main()
