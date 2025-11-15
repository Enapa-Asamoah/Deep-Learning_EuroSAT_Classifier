# Contributing to Deep Learning Research Projects

Thank you for your interest in contributing to this research repository! This document provides guidelines for contributing to the projects.

## Project Structure

This repository contains four independent research projects:
- **LightVision**: Lightweight CNNs for image classification
- **FairVoice**: Bias and explainability in speech emotion recognition
- **MultiSense**: Multimodal emotion understanding
- **VisionXplain**: Interpretable Vision Transformers for medical imaging

Each project is managed independently. Please refer to the project-specific README and IMPLEMENTATION.md files for detailed information.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:
1. Check if the issue already exists in the project's issue tracker
2. Create a new issue with:
   - Clear description of the problem or suggestion
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**:
   - Follow the project's code style
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass
4. **Commit your changes**:
   ```bash
   git commit -m "Description of your changes"
   ```
5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request** with a clear description of your changes

## Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and modular
- Add comments for complex logic

## Testing

- Write unit tests for new functionality
- Ensure all existing tests pass
- Aim for high test coverage
- Test edge cases and error conditions

## Documentation

- Update README.md if adding new features
- Add docstrings to all functions and classes
- Update IMPLEMENTATION.md if changing the workflow
- Include examples in documentation

## Research Contributions

For research-related contributions:
- Ensure reproducibility (fixed seeds, versioned datasets)
- Include statistical analysis where applicable
- Document experimental setup clearly
- Provide ablation studies for major changes

## Important Notes

- **Medical Data**: For VisionXplain, ensure all medical data handling complies with HIPAA/GDPR
- **Bias & Fairness**: For FairVoice, consider ethical implications of changes
- **Reproducibility**: Maintain reproducibility standards (fixed seeds, versioned configs)
- **Licensing**: Ensure any new dependencies are compatible with the project license

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

## Recognition

Contributors will be recognized in:
- Project README files
- Publication acknowledgments (where applicable)
- Release notes

Thank you for contributing to open research!

