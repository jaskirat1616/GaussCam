# Contributing to GaussCam

Thank you for your interest in contributing to GaussCam! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/jaskirat1616/GaussCam/issues)
2. If not, create a new issue using the bug report template
3. Provide as much detail as possible:
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, GPU, Python version)
   - Error logs and screenshots

### Suggesting Features

1. Check if the feature has already been suggested in [Issues](https://github.com/jaskirat1616/GaussCam/issues)
2. If not, create a new issue using the feature request template
3. Describe the feature clearly and explain its use case

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the code style guidelines
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for classes and functions
- Keep functions focused and small
- Add comments for complex logic

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/jaskirat1616/GaussCam.git
   cd GaussCam
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install PyTorch (platform-specific):
   - **Windows (CUDA)**: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
   - **macOS (MPS)**: `pip install torch torchvision`

## Testing

Run tests with:
```bash
python -m pytest backend/tests/
```

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## Questions?

Feel free to open an issue for any questions or discussions!

