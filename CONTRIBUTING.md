# Contributing to encouRAGe   <img src="assets/logo.png" alt="Encourage Logo" width="32" /><br /> 

Thanks for your interest in contributing! 🚀  
We welcome all kinds of contributions—bug reports, feature suggestions, and pull requests.

## 🛠 How to Contribute

- Fork the repo and create your branch: `git checkout -b feature/your-feature`
- Make your changes
- Ensure all tests pass
- Commit with a clear message: `git commit -m "Add feature X"`
- Push and open a Pull Request

## 🧪 Running Tests

```bash
# Install dependencies
uv sync

# Run tests
make tests

# Run linting
make lint
```

## ✏️ Code Style

You will find all settings for the linter in the `pyproject.toml` file. Please ensure that your code adheres to the following guidelines:

- Format with `black` and `isort` (`isort` profile: `black`)
- Lint and check code quality with `ruff`:
  - Line length: `100`
  - Selected rules: `E`, `F`, `W`, `I`, `D`, `A`, `N`, `B`, `SIM`, `C4`, `TID`
  - Ignored rules include:
    - `E741`, `D213`, `D105`, `D107`, `D203`, `D401`, `D407`, `D406`, `D106`
    - `B006`, `B008`, `B905`
  - Jupyter notebooks (`*.ipynb`) are excluded from formatting
  - `test_*.py`: ignores `D`, `E402`
  - `__init__.py`: ignores `D`
  - Relative imports are **not allowed**
  


## 🐛 Bug Reports & Feature Requests

- Use GitHub Issues
- Provide steps to reproduce (for bugs)
- Describe expected behavior

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you! ❤️
