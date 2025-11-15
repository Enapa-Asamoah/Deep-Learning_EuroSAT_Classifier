# Repository Setup Guide

This document provides a quick setup guide for the Deep Learning Research Projects repository.

##  Repository Overview

This repository contains four independent research projects:

1. **LightVision** - Lightweight CNNs for image classification
2. **FairVoice** - Bias and explainability in speech emotion recognition
3. **MultiSense** - Multimodal emotion understanding
4. **VisionXplain** - Interpretable Vision Transformers for medical imaging

##  Initial Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Deep Learning"
```

### 2. Set Up Python Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or use conda
conda create -n dl-projects python=3.10
conda activate dl-projects
```

### 3. Install Project Dependencies

Each project has its own requirements file. Install dependencies for the project you want to work on:

```bash
# For LightVision
cd lightvision
pip install -r requirements.txt

# For FairVoice
cd fairvoice
pip install -r requirements.txt

# For MultiSense
cd multisense
pip install -r requirements.txt

# For VisionXplain
cd visionxplain
pip install -r requirements.txt
```

##  Project Structure

Each project follows a consistent structure:

```
project_name/
├── src/              # Source code
├── configs/          # Configuration files (YAML)
├── data/             # Dataset storage
├── experiments/      # Experiment tracking
├── notebooks/        # Jupyter notebooks
├── tests/            # Unit and integration tests
├── docs/             # Documentation
├── scripts/          # Standalone scripts
├── outputs/          # Model outputs, logs, plots
├── README.md         # Project overview
├── IMPLEMENTATION.md # Detailed implementation guide
└── requirements.txt  # Python dependencies
```

##  Development Setup

### 1. Install Development Tools

```bash
pip install black flake8 mypy pytest pytest-cov
```

### 2. Set Up Experiment Tracking

For projects using Weights & Biases:

```bash
wandb login
```

For projects using MLflow:

```bash
# MLflow runs locally by default
# No additional setup needed
```

### 3. Configure Data Directories

Create data directories and download datasets:

```bash
# Example for LightVision
cd lightvision
mkdir -p data/raw data/processed
# Download EuroSAT or TrashNet dataset to data/raw/
```

##  Getting Started with a Project

1. **Read the Project README**: Start with `project_name/README.md`
2. **Review Implementation Guide**: Check `project_name/IMPLEMENTATION.md`
3. **Check Configuration**: Review `project_name/configs/` for available configurations
4. **Run Preprocessing**: Follow the preprocessing scripts in `project_name/scripts/preprocessing/`
5. **Start Training**: Use the training scripts in `project_name/scripts/training/`

## 🧪 Running Tests

```bash
# Run all tests for a project
cd project_name
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

##  Experiment Tracking

- **Weights & Biases**: Projects use W&B for experiment tracking
- **MLflow**: Alternative tracking system
- **TensorBoard**: For training visualization

## 🔍 Code Quality

- **Formatting**: Use `black` for code formatting
- **Linting**: Use `flake8` for code linting
- **Type Checking**: Use `mypy` for type checking

```bash
# Format code
black src/ scripts/

# Lint code
flake8 src/ scripts/

# Type check
mypy src/
```

## 📚 Documentation

- Each project has comprehensive README and IMPLEMENTATION guides
- API documentation can be generated using Sphinx
- See `docs/` directory in each project for additional documentation

##  Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

##  Important Notes

- **Data Privacy**: For VisionXplain, ensure medical data handling complies with HIPAA/GDPR
- **Reproducibility**: Always use fixed seeds for experiments
- **Version Control**: Use DVC for large datasets
- **Resource Requirements**: Check project READMEs for GPU/CPU requirements

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the correct virtual environment
2. **CUDA Errors**: Check PyTorch CUDA installation
3. **Dataset Not Found**: Verify data directory structure
4. **Memory Issues**: Reduce batch size in config files

### Getting Help

- Check project-specific README and IMPLEMENTATION files
- Review issue tracker
- Contact project maintainers

##  License

See [LICENSE](LICENSE) for license information.

