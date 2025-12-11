# LightVision: Lightweight CNNs for Real-World Image Classification

## Project Overview

**LightVision** is a research project that addresses the challenge of deploying accurate image classification models in low-resource environments. While modern deep learning models achieve good performance, they often require substantial computational resources that make them impractical for edge devices, mobile applications, or resource-constrained settings.

This project compares model compression techniques—knowledge distillation (KD), quantization-aware training (QAT), and pruning—on realistic small-scale datasets. By evaluating these techniques individually and in combination, LightVision provides insights for practitioners deploying vision models in production environments.

## Key Innovations

- **Compression Framework**: Evaluation of KD, QAT, and pruning techniques
- **Statistical Analysis**: Comparisons with multiple runs and significance testing
- **Real-World Datasets**: Evaluation on EuroSAT (RGB) or TrashNet
- **Hardware Benchmarking**: CPU latency, energy consumption, and model size measurements
- **Deployment Support**: Models for Raspberry Pi and Android devices
- **Reproducibility**: Complete workflow with fixed seeds, versioned datasets, and documentation

## Project Goals

1. **Implement and train** baseline teacher (high-capacity) and student (lightweight) models
2. **Apply compression techniques** including knowledge distillation, QAT, and pruning (structured/unstructured) separately and in combinations
3. **Measure accuracy vs efficiency** trade-offs: model size, FLOPs, CPU latency, and energy consumption
4. **Demonstrate deployment** on target devices (Raspberry Pi or Android via PyTorch Mobile / TFLite)
5. **Produce reproducible artifacts** including code, dataset splits, and a publication-ready report with ablations and statistical tests

## Project Structure

```
lightvision/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures (teacher, student, compressed)
│   ├── training/          # Training scripts and loops
│   ├── compression/       # Compression techniques
│   │   ├── distillation/  # Knowledge distillation
│   │   ├── quantization/  # QAT and post-training quantization
│   │   └── pruning/       # Structured and unstructured pruning
│   ├── evaluation/        # Evaluation metrics and benchmarking
│   ├── deployment/        # Deployment utilities and conversion
│   └── utils/             # Utility functions
├── configs/               # Configuration files (YAML)
│   ├── baseline_config.yaml
│   ├── distillation_config.yaml
│   ├── qat_config.yaml
│   └── pruning_config.yaml
├── experiments/           # Experiment tracking and results
│   ├── baseline/         # Baseline model experiments
│   ├── distilled/        # Knowledge distillation experiments
│   ├── quantized/        # Quantization experiments
│   └── pruned/           # Pruning experiments
├── data/                  # Dataset storage
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed images
│   └── splits/           # Fixed train/val/test splits
├── notebooks/             # Jupyter notebooks
│   ├── exploration/      # Data exploration
│   ├── analysis/         # Results analysis
│   └── interpretability/ # Model interpretability
├── tests/                 # Unit and integration tests
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── docs/                  # Documentation
│   ├── paper/            # Research paper drafts
│   ├── api/              # API documentation
│   └── deployment/       # Deployment guides
├── scripts/               # Standalone scripts
│   ├── preprocessing/    # Data preprocessing
│   ├── training/         # Model training
│   ├── evaluation/       # Evaluation and benchmarking
│   └── deployment/       # Deployment scripts
├── outputs/               # Model outputs, logs, plots
│   ├── models/           # Trained models
│   ├── logs/             # Training logs
│   ├── plots/            # Visualizations
│   └── reports/          # Generated reports
└── deployment/           # Deployment configurations
    ├── raspberry_pi/     # Raspberry Pi deployment
    └── android/          # Android deployment
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. Download dataset (EuroSAT RGB or TrashNet)
2. Run preprocessing pipeline:
```bash
python scripts/preprocessing/01_prepare_data.py
```

### Training Baseline Teacher Model

```bash
python scripts/training/02_train_teacher_model.py \
  --epochs 20 \
  --batch_size 32 \
  --lr 0.001
```

### Training Student Models

```bash
# Lightweight student model
python scripts/training/03_train_student_model.py \
  --epochs 20 \
  --batch_size 32 \
  --lr 0.001
```

### Model Compression

```bash
# Knowledge Distillation, Pruning and Quantization
python scripts/training/04_compress_model.py \
  --student_arch student_resnet18 \
  --student_checkpoint outputs/models/student_resnet18.pth \
  --methods distillation pruning qat
```

### Evaluation and Benchmarking

```bash
# Comprehensive evaluation
python scripts/evaluation/05_evaluate_model.py
```

## Datasets

### EuroSAT (RGB)
- **Description**: Sentinel-2 satellite images for land use classification
- **Classes**: 10 land use categories
- **Size**: ~27,000 labeled images
- **Image Size**: 64×64 pixels
- **Download**: [EuroSAT Dataset](https://github.com/phelber/EuroSAT)

### TrashNet
- **Description**: Images of recyclable materials for waste classification
- **Classes**: 6 material categories
- **Size**: ~2,500 images
- **Image Size**: Variable (resized to 224×224)
- **Download**: [TrashNet Dataset](https://github.com/garythung/trashnet)

## Research Contributions

This project contributes to the field through:

1. **Empirical Comparison**: Rigorous statistical comparison of compression techniques on realistic datasets
2. **Combined Strategies**: Novel evaluation of combined compression pipelines
3. **Hardware Benchmarks**: Real-world deployment metrics (latency, energy, model size)
4. **Reproducibility**: Complete workflow with fixed seeds and versioned artifacts
5. **Practical Insights**: Actionable recommendations for practitioners

## Expected Deliverables

-  Trained teacher and student models (baseline and compressed)
-  Comprehensive comparison tables (accuracy, FLOPs, model size, latency, energy)
-  Statistical analysis with significance tests
-  Ablation studies on compression techniques
-  Deployment artifacts (ONNX, TFLite, PyTorch Mobile)
-  Publication-ready technical report (6-8 pages)
-  Reproducibility package (code, configs, dataset splits)

## Publication Readiness

This project is designed to produce a high-impact publication with:

- **Novel Contributions**: Empirical comparison of compression techniques on small-scale datasets
- **Statistical Rigor**: Multiple runs, significance testing, confidence intervals
- **Comprehensive Evaluation**: Accuracy, efficiency, and deployment metrics
- **Reproducibility**: Complete codebase with fixed seeds and documentation
- **Practical Relevance**: Real-world datasets and deployment scenarios

## Contributing

This is a research project. For questions or contributions, please refer to the implementation guide.

## License

[Specify license]

## Acknowledgments

- EuroSAT and TrashNet dataset creators
- PyTorch and TensorFlow communities
- ONNX Runtime and TFLite teams
- Open-source compression research community