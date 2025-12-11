# LightVision Implementation Guide

## Implementation Roadmap

This document provides a practical implementation guide for the LightVision project. The guide is organized to help you produce reproducible results with proper statistical analysis.

## Phase 1: Foundation & Baseline Models

### 1.1 Dataset Selection and Preparation

**Objective**: Establish a robust, reproducible data pipeline with fixed splits

**Tasks**:
- [ ] **Dataset Selection** (`scripts/preprocessing/select_dataset.py`)
  - Choose between EuroSAT (RGB) or TrashNet
  - Document dataset characteristics (size, classes, image dimensions)
  - Download and verify dataset integrity
  - Create dataset metadata file

- [ ] **Data Preprocessing** (`src/data/preprocessing.py`)
  - Image normalization (ImageNet statistics or dataset-specific)
  - Resize to consistent dimensions (224×224 for TrashNet, 64×64 for EuroSAT)
  - Data augmentation pipeline:
    - Training: Random crop, horizontal flip, color jitter, rotation
    - Validation/Test: Center crop only (no augmentation)
  - Create fixed train/val/test splits (70/15/15) with random seed 42
  - Save split indices for reproducibility

- [ ] **Data Loaders** (`src/data/dataloader.py`)
  - PyTorch DataLoader with proper batching
  - Stratified sampling to ensure class balance
  - Caching mechanism for processed images
  - Support for multiple workers and pin memory

- [ ] **Dataset Statistics** (`notebooks/exploration/dataset_analysis.ipynb`)
  - Class distribution analysis
  - Image quality assessment
  - Visualization of sample images per class
  - Dataset bias analysis

**Key Files**:
- `src/data/preprocessing.py`
- `src/data/dataloader.py`
- `src/data/dataset_manager.py`
- `scripts/preprocessing/prepare_data.py`
- `data/splits/train_val_test_split_seed42.json`

**Deliverables**:
- Preprocessed dataset with fixed splits
- Dataset statistics report
- Data loading pipeline with caching

### 1.2 Baseline Teacher Model

**Objective**: Train a high-capacity teacher model that serves as the knowledge source

**Tasks**:
- [ ] **Teacher Architecture** (`src/models/teacher.py`)
  - High-capacity CNN: ResNet-50, ResNet-101, or EfficientNet-B3
  - Pre-trained on ImageNet (transfer learning)
  - Custom classification head for target dataset
  - Architecture documentation with parameter counts

- [ ] **Training Pipeline** (`src/training/trainer.py`)
  - Standard training loop with:
    - Cross-entropy loss
    - AdamW optimizer with cosine annealing
    - Learning rate: 1e-4 with warmup
    - Weight decay: 1e-4
    - Batch size: 32-64 (depending on GPU memory)
  - Early stopping with patience (monitor validation accuracy)
  - Model checkpointing (best model + last epoch)
  - Training curves logging (loss, accuracy per epoch)

- [ ] **Evaluation Framework** (`src/evaluation/evaluator.py`)
  - Metrics: Accuracy, Top-5 accuracy, Per-class F1-score
  - Confusion matrix generation
  - Classification report
  - Inference time measurement

- [ ] **Reproducibility** (`configs/baseline_config.yaml`)
  - Fixed random seeds (42, 123, 456, 789, 1011)
  - Deterministic operations (torch.backends.cudnn.deterministic = True)
  - Experiment configuration in YAML
  - Logging of all hyperparameters

**Key Files**:
- `src/models/teacher.py`
- `src/training/trainer.py`
- `src/evaluation/evaluator.py`
- `scripts/training/train_baseline.py`
- `configs/baseline_config.yaml`

**Deliverables**:
- Trained teacher model (checkpoint)
- Training logs and curves
- Baseline performance metrics
- Reproducibility documentation

### 1.3 Baseline Student Models

**Objective**: Design and train lightweight student models without compression

**Tasks**:
- [ ] **Student Architectures** (`src/models/student.py`)
  - Multiple lightweight architectures:
    - MobileNetV2 (baseline student)
    - MobileNetV3 (efficient variant)
    - ShuffleNetV2 (channel shuffling)
    - Custom lightweight CNN (3-4 layers)
  - Document parameter counts and FLOPs for each
  - Pre-trained on ImageNet when available

- [ ] **Student Training** (`src/training/student_trainer.py`)
  - Train each student architecture independently
  - Same training protocol as teacher (for fair comparison)
  - Record baseline accuracy for each student
  - Measure model size and FLOPs

- [ ] **Baseline Comparison** (`notebooks/analysis/baseline_comparison.ipynb`)
  - Teacher vs student accuracy comparison
  - Model size vs accuracy trade-off
  - FLOPs vs accuracy analysis
  - Select best student architecture for compression experiments

**Key Files**:
- `src/models/student.py`
- `src/training/student_trainer.py`
- `scripts/training/train_student.py`

**Deliverables**:
- Trained student models
- Baseline comparison report
- Selected student architecture for compression

## Phase 2: Knowledge Distillation

### 2.1 Distillation Framework

**Objective**: Implement and evaluate knowledge distillation from teacher to student

**Tasks**:
- [ ] **Distillation Architecture** (`src/compression/distillation/kd_framework.py`)
  - Teacher-student pair setup
  - Soft target generation (teacher logits)
  - Temperature scaling (T = 3, 5, 7 - hyperparameter)
  - Combined loss function:
    - KL divergence for soft targets
    - Cross-entropy for hard targets
    - Weighted combination: α * soft_loss + (1-α) * hard_loss

- [ ] **Distillation Training** (`src/training/distillation_trainer.py`)
  - Freeze teacher model (inference only)
  - Train student with distillation loss
  - Hyperparameter search:
    - Temperature: [3, 5, 7]
    - Alpha (soft/hard weight): [0.3, 0.5, 0.7]
  - Learning rate: 1e-4 (may need adjustment)
  - Training for same number of epochs as baseline

- [ ] **Evaluation** (`src/evaluation/distillation_evaluator.py`)
  - Compare distilled student vs baseline student
  - Measure accuracy improvement
  - Analyze knowledge transfer effectiveness
  - Visualize attention/feature maps (if applicable)

**Key Files**:
- `src/compression/distillation/kd_framework.py`
- `src/training/distillation_trainer.py`
- `scripts/training/train_distilled.py`
- `configs/distillation_config.yaml`

**Deliverables**:
- Distilled student models
- Hyperparameter analysis
- Distillation effectiveness report

## Phase 3: Quantization-Aware Training

### 3.1 QAT Implementation

**Objective**: Implement quantization-aware training for INT8 models

**Tasks**:
- [ ] **QAT Framework** (`src/compression/quantization/qat.py`)
  - Fake quantization layers (simulate INT8 during training)
  - Quantization scheme: Symmetric or Asymmetric
  - Per-channel quantization for weights
  - Per-tensor quantization for activations
  - Calibration dataset preparation

- [ ] **QAT Training** (`src/training/qat_trainer.py`)
  - Start from pre-trained student model
  - Fine-tune with fake quantization
  - Learning rate: 1e-5 (lower than baseline)
  - Training for fewer epochs (fine-tuning)
  - Monitor quantization-aware accuracy

- [ ] **Post-Training Quantization** (`src/compression/quantization/ptq.py`)
  - Static quantization (calibration-based)
  - Dynamic quantization (runtime)
  - Compare PTQ vs QAT performance
  - Measure accuracy drop from FP32

- [ ] **Quantization Evaluation** (`src/evaluation/quantization_evaluator.py`)
  - Accuracy comparison: FP32 vs INT8
  - Model size reduction (4x for INT8)
  - Inference speedup measurement
  - Energy consumption (if hardware available)

**Key Files**:
- `src/compression/quantization/qat.py`
- `src/compression/quantization/ptq.py`
- `src/training/qat_trainer.py`
- `scripts/training/train_qat.py`
- `configs/qat_config.yaml`

**Deliverables**:
- Quantized models (QAT and PTQ)
- Quantization analysis report
- Performance vs accuracy trade-offs

## Phase 4: Pruning

### 4.1 Pruning Strategies

**Objective**: Implement structured and unstructured pruning

**Tasks**:
- [ ] **Unstructured Pruning** (`src/compression/pruning/unstructured.py`)
  - Magnitude-based pruning (L1/L2 norm)
  - Iterative pruning schedule:
    - Prune 20% → fine-tune → prune 20% → fine-tune
    - Target sparsity: 50%, 70%, 90%
  - Global vs layer-wise pruning
  - Pruning mask generation and application

- [ ] **Structured Pruning** (`src/compression/pruning/structured.py`)
  - Channel pruning (remove entire filters)
  - Layer pruning (remove entire layers)
  - Filter importance scoring (L1 norm, gradient-based)
  - Architecture modification after pruning

- [ ] **Pruning Training** (`src/training/pruning_trainer.py`)
  - Iterative pruning and fine-tuning
  - Learning rate: 1e-5 (fine-tuning rate)
  - Pruning schedule configuration
  - Recovery training after each pruning step

- [ ] **Pruning Evaluation** (`src/evaluation/pruning_evaluator.py`)
  - Accuracy vs sparsity curves
  - Model size reduction
  - FLOPs reduction
  - Inference speedup (actual vs theoretical)

**Key Files**:
- `src/compression/pruning/unstructured.py`
- `src/compression/pruning/structured.py`
- `src/training/pruning_trainer.py`
- `scripts/training/train_pruned.py`
- `configs/pruning_config.yaml`

**Deliverables**:
- Pruned models at various sparsity levels
- Pruning analysis report
- Sparsity vs accuracy trade-offs

## Phase 5: Combined Compression

### 5.1 Combined Pipeline

**Objective**: Evaluate compression techniques in combination

**Tasks**:
- [ ] **Combination Strategies** (`src/compression/combined/pipeline.py`)
  - Strategy 1: Pruning → Distillation → Quantization
  - Strategy 2: Distillation → Pruning → Quantization
  - Strategy 3: Distillation → Quantization → Pruning
  - Evaluate all combinations systematically

- [ ] **Combined Training** (`src/training/combined_trainer.py`)
  - Sequential application of compression techniques
  - Fine-tuning after each compression step
  - Monitor cumulative accuracy drop
  - Record intermediate checkpoints

- [ ] **Combined Evaluation** (`src/evaluation/combined_evaluator.py`)
  - Compare individual vs combined compression
  - Analyze synergy effects
  - Measure final model characteristics
  - Pareto frontier analysis

**Key Files**:
- `src/compression/combined/pipeline.py`
- `src/training/combined_trainer.py`
- `scripts/training/train_combined.py`
- `configs/combined_config.yaml`

**Deliverables**:
- Combined compression models
- Combination analysis report
- Best compression pipeline identification

## Phase 6: Comprehensive Evaluation

### 6.1 Metrics Collection

**Objective**: Collect comprehensive metrics for all models

**Tasks**:
- [ ] **Accuracy Metrics** (`src/evaluation/accuracy_metrics.py`)
  - Top-1 and Top-5 accuracy
  - Per-class precision, recall, F1-score
  - Confusion matrices
  - Statistical significance testing (t-tests, bootstrap)

- [ ] **Efficiency Metrics** (`src/evaluation/efficiency_metrics.py`)
  - Model size (MB, parameters)
  - FLOPs calculation (using thop or fvcore)
  - Memory footprint
  - Inference latency (CPU, single-threaded)
  - Throughput (images/second)

- [ ] **Energy Consumption** (`src/evaluation/energy_metrics.py`)
  - CPU energy (if hardware monitoring available)
  - Power profiling
  - Energy per inference

- [ ] **Robustness Tests** (`src/evaluation/robustness.py`)
  - Adversarial robustness (optional)
  - Corruption robustness (noise, blur)
  - Cross-dataset evaluation (if applicable)

**Key Files**:
- `src/evaluation/accuracy_metrics.py`
- `src/evaluation/efficiency_metrics.py`
- `src/evaluation/energy_metrics.py`
- `scripts/evaluation/benchmark.py`

**Deliverables**:
- Comprehensive metrics table
- Statistical analysis results
- Performance comparison plots

### 6.2 Statistical Analysis

**Objective**: Perform rigorous statistical analysis for publication

**Tasks**:
- [ ] **Multiple Runs** (`scripts/evaluation/run_multiple_experiments.py`)
  - Run each experiment 5 times with different seeds
  - Record mean and standard deviation
  - Calculate confidence intervals (95%)

- [ ] **Significance Testing** (`src/evaluation/statistical_tests.py`)
  - Paired t-tests between methods
  - Bonferroni correction for multiple comparisons
  - Effect size calculation (Cohen's d)
  - Statistical power analysis

- [ ] **Visualization** (`notebooks/analysis/statistical_analysis.ipynb`)
  - Box plots with confidence intervals
  - Statistical significance annotations
  - Trade-off curves (accuracy vs efficiency)
  - Pareto frontier plots

**Key Files**:
- `src/evaluation/statistical_tests.py`
- `scripts/evaluation/run_multiple_experiments.py`
- `notebooks/analysis/statistical_analysis.ipynb`

**Deliverables**:
- Statistical test results
- Publication-ready plots
- Significance analysis report

## Phase 7: Deployment

### 7.1 Model Conversion

**Objective**: Convert models to deployment formats

**Tasks**:
- [ ] **ONNX Conversion** (`src/deployment/onnx_converter.py`)
  - Convert PyTorch models to ONNX
  - Verify ONNX model correctness
  - Optimize ONNX graph
  - Test ONNX inference

- [ ] **TensorFlow Lite Conversion** (`src/deployment/tflite_converter.py`)
  - Convert to TFLite format
  - Apply quantization (if not already quantized)
  - Optimize for mobile deployment
  - Test TFLite inference

- [ ] **PyTorch Mobile** (`src/deployment/pytorch_mobile.py`)
  - Convert to TorchScript
  - Optimize for mobile
  - Quantize for mobile (if applicable)

**Key Files**:
- `src/deployment/onnx_converter.py`
- `src/deployment/tflite_converter.py`
- `src/deployment/pytorch_mobile.py`
- `scripts/deployment/convert_models.py`

**Deliverables**:
- Deployment-ready models (ONNX, TFLite)
- Conversion verification reports

### 7.2 Hardware Benchmarking

**Objective**: Benchmark models on target hardware

**Tasks**:
- [ ] **Raspberry Pi Benchmarking** (`deployment/raspberry_pi/benchmark.py`)
  - Deploy ONNX or TFLite model
  - Measure inference latency
  - Measure memory usage
  - Batch inference testing
  - Energy consumption (if possible)

- [ ] **Android Benchmarking** (`deployment/android/benchmark.py`)
  - Deploy TFLite model
  - Measure inference on mobile device
  - Test with different batch sizes
  - Real-time performance testing

- [ ] **Benchmarking Report** (`scripts/evaluation/benchmark_hardware.py`)
  - Compare all models on hardware
  - Latency vs accuracy trade-offs
  - Hardware-specific recommendations

**Key Files**:
- `deployment/raspberry_pi/benchmark.py`
- `deployment/android/benchmark.py`
- `scripts/evaluation/benchmark_hardware.py`

**Deliverables**:
- Hardware benchmarking results
- Deployment performance report
- Device-specific recommendations


## Phase 8: Documentation & Paper Writing

### 8.1 Reproducibility Package

**Objective**: Ensure complete reproducibility

**Tasks**:
- [ ] **Code Documentation** (`docs/`)
  - API documentation (Sphinx)
  - Code comments and docstrings
  - Usage examples
  - Troubleshooting guide

- [ ] **Configuration Files** (`configs/`)
  - All experiment configurations
  - Hyperparameter documentation
  - Seed documentation

- [ ] **Dataset Splits** (`data/splits/`)
  - Fixed train/val/test splits
  - Split indices for reproducibility
  - Dataset versioning

- [ ] **Environment Setup** (`docs/setup.md`)
  - Python version
  - Package versions (requirements.txt)
  - Hardware requirements
  - Step-by-step setup guide

- [ ] **Reproduction Script** (`scripts/reproduce_experiments.sh`)
  - Single script to reproduce all experiments
  - Automated pipeline
  - Verification steps

**Key Files**:
- `docs/setup.md`
- `docs/api/`
- `scripts/reproduce_experiments.sh`
- `REPRODUCIBILITY.md`

**Deliverables**:
- Complete reproducibility package
- Setup documentation
- Reproduction verification

### 9.2 Paper Writing

**Objective**: Write publication-ready paper

**Tasks**:
- [ ] **Paper Structure** (`docs/paper/`)
  - Abstract (150-200 words)
  - Introduction and motivation
  - Related work
  - Methodology (detailed)
  - Experiments and results
  - Ablation studies
  - Discussion and conclusions
  - Ethics statement (if applicable)

- [ ] **Figures and Tables** (`docs/paper/figures/`)
  - Performance comparison tables
  - Trade-off curves
  - Ablation study plots
  - Architecture diagrams
  - Statistical significance plots

- [ ] **Results Presentation** (`docs/paper/results/`)
  - Main results table
  - Statistical test results
  - Hardware benchmarking tables
  - Ablation study tables

**Key Files**:
- `docs/paper/main.tex` or `docs/paper/main.md`
- `docs/paper/figures/`
- `docs/paper/tables/`

**Deliverables**:
- Publication-ready paper (6-8 pages)
- All figures and tables
- Supplementary material

## Technical Implementation Details

### Model Architectures

**Teacher Models**:
- ResNet-50: ~25M parameters, baseline accuracy target >90%
- ResNet-101: ~44M parameters, higher capacity option
- EfficientNet-B3: ~12M parameters, efficient architecture

**Student Models**:
- MobileNetV2: ~3.4M parameters, baseline student
- MobileNetV3: ~4.2M parameters, improved efficiency
- ShuffleNetV2: ~2.3M parameters, channel shuffling
- Custom Lightweight CNN: ~500K-1M parameters, minimal design

### Compression Targets

- **Model Size**: Reduce to <5MB (from ~10-50MB)
- **FLOPs**: Reduce to <100M FLOPs (from ~500M-1B)
- **Latency**: <50ms on Raspberry Pi, <20ms on mobile
- **Accuracy Retention**: >90% of baseline accuracy

### Evaluation Protocol

1. **Fixed Splits**: 70% train, 15% validation, 15% test (seed 42)
2. **Multiple Runs**: 5 runs per experiment with different seeds
3. **Statistical Testing**: Paired t-tests with Bonferroni correction
4. **Metrics**: Accuracy, F1-score, model size, FLOPs, latency, energy
5. **Reporting**: Mean ± std with 95% confidence intervals

### Statistical Rigor

- **Multiple Seeds**: 5 independent runs per configuration
- **Significance Testing**: p < 0.05 with correction
- **Effect Sizes**: Cohen's d for practical significance
- **Confidence Intervals**: 95% CI for all metrics
- **Reproducibility**: Fixed seeds, deterministic operations

## Success Metrics

- **Accuracy**: Maintain >85% accuracy with <5MB model
- **Compression**: Achieve 10x compression with <5% accuracy drop
- **Latency**: <50ms inference on Raspberry Pi
- **Statistical Significance**: Significant improvements over baselines
- **Reproducibility**: All experiments reproducible with provided code

## Next Steps

1. Set up development environment
2. Download and preprocess dataset
3. Train baseline teacher and student models
4. Implement and evaluate compression techniques
5. Perform comprehensive evaluation and statistical analysis
6. Deploy models and benchmark on hardware
7. Conduct ablation studies
8. Write paper and prepare reproducibility package

## Publication Checklist

- [ ] All experiments run with 5 different seeds
- [ ] Statistical significance tests performed
- [ ] Confidence intervals reported
- [ ] Ablation studies completed
- [ ] Hardware benchmarking done
- [ ] Reproducibility package prepared
- [ ] Code fully documented
- [ ] Paper written and reviewed
- [ ] All figures and tables generated
- [ ] Supplementary material prepared

