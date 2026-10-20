# Contributing to Battery Cathode ML

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## 🌟 How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include as much detail as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** to demonstrate the steps
- **Describe the behavior you observed** after following the steps
- **Explain which behavior you expected** instead
- **Include screenshots or code snippets** if applicable
- **Specify your environment** (OS, Python version, etc.)

#### Bug Report Template
```markdown
**Description**: Short and descriptive explanation of the bug

**Steps to Reproduce**:
1. Step one
2. Step two
3. Step three

**Expected Behavior**: What should happen

**Actual Behavior**: What actually happened

**Environment**:
- OS: 
- Python version: 
- Package versions: 
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples** to demonstrate the steps
- **Describe the current behavior** and **explain why** you'd like to see it changed
- **Explain why this enhancement would be useful** to other users
- **List some similar enhancements** in other projects (if applicable)

### Pull Requests

**Please follow these steps for pull requests:**

1. **Fork the repository** and create your branch from `main`
2. **Add tests** for new features
3. **Ensure test suite passes**
4. **Make sure your code follows the style guidelines**
5. **Update documentation** as necessary
6. **Write a descriptive commit message**
7. **Reference related issues** in your PR description

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ilo-love7/battery-cathode-ml.git
   cd battery-cathode-ml
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. **Set up pre-commit hooks** (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Code Style Guidelines

- **Follow PEP 8** for Python code style
- **Use type hints** where possible
- **Write docstrings** for all functions and classes
- **Keep functions small and focused** (Single Responsibility Principle)
- **Write meaningful commit messages**:
  - Start with a verb in imperative mood
  - Keep the first line under 50 characters
  - Add detailed description in body if needed

Example:
```
feat: add voltage prediction model training

- Implement train_voltage.py script
- Add feature engineering for voltage prediction
- Update README with new usage instructions
```

### Testing

- Write tests for new features in the `tests/` directory
- Ensure all existing tests pass before submitting PR
- Aim for high test coverage

Run tests:
```bash
pytest tests/ -v
pytest tests/ --cov=src battery_cathode_ml  # With coverage
```

### Documentation

- Update `README.md` if you add new features
- Add docstrings to all public functions and classes
- Update API documentation if applicable
- Include examples for complex features

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Build process or auxiliary tool changes

Examples:
```
feat(voltage): add voltage prediction model
fix(formula): correct parsing error for nested brackets
docs(readme): update installation instructions
test(features): add unit tests for feature extraction
```

## 🔧 Areas We Need Help

We welcome contributions in many areas:

- **Feature Implementation**: Add new prediction models or features
- **Bug Fixes**: Fix reported issues
- **Documentation**: Improve docs, add examples, fix typos
- **Testing**: Increase test coverage
- **Performance**: Optimize existing code
- **UI/UX**: Improve visualization quality
- **Data**: Add new datasets or improve data processing
- **Integration**: Support additional ML frameworks

## 💬 Questions?

Feel free to open an issue with questions or reach out to maintainers.

## 📝 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Battery Cathode ML! 🎉
