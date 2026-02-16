# Contributing

We welcome contributions to `synth-pdb`! Whether you're fixing bugs, adding new features, improving documentation, or suggesting enhancements, your input helps make this project better for everyone. This guide will help you get started.

## Ways to Contribute

There are many ways to contribute to `synth-pdb`:

*   **Report Bugs:** If you find a bug, please open an issue on our [GitHub repository](https://github.com/elkins/synth-pdb/issues). Provide clear steps to reproduce the bug, expected behavior, and actual behavior.
*   **Suggest Features:** Have an idea for a new feature or improvement? Open an issue on GitHub to discuss it.
*   **Improve Documentation:** Spotted a typo, an unclear explanation, or want to add a new example? Documentation contributions are highly valued!
*   **Write Code:** Implement new features, fix bugs, or refactor existing code.
*   **Provide Feedback:** Share your thoughts on usability, performance, or anything else.

## Getting Started with Code Contributions

If you plan to contribute code, please follow these steps:

### 1. Fork the Repository

First, fork the `synth-pdb` repository on GitHub to your personal account.

### 2. Clone Your Fork

Clone your forked repository to your local machine:

```bash
git clone https://github.com/YOUR_USERNAME/synth-pdb.git
cd synth-pdb
```

### 3. Set Up Your Development Environment

It's recommended to work in a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
pip install -e .           # Install synth-pdb in editable mode
pip install -r requirements-dev.txt # Install development dependencies
```

### 4. Create a New Branch

Create a new branch for your feature or bug fix:

```bash
git checkout -b feature/your-feature-name-or-bugfix/your-bugfix-name
```

Choose a descriptive branch name.

### 5. Make Your Changes

Implement your changes. Please adhere to the existing coding style and conventions.

### 6. Test Your Changes

Ensure your changes work as expected and don't introduce new bugs. If you're adding new features, please write unit tests for them.

To run tests:

```bash
pytest
```

### 7. Lint and Format Your Code

We use `ruff` and `black` for linting and formatting. Please run them before committing:

```bash
ruff check . --fix
black .
```

### 8. Commit Your Changes

Write clear and concise commit messages. A good commit message explains *what* changed and *why*.

```bash
git add .
git commit -m "feat: Add new feature for X"
```

### 9. Push Your Branch

```bash
git push origin feature/your-feature-name
```

### 10. Open a Pull Request

Go to your forked repository on GitHub and open a Pull Request (PR) to the `main` branch of the original `synth-pdb` repository. Please provide a clear description of your changes and reference any related issues.

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project, you agree to abide by its terms.

Thank you for contributing to `synth-pdb`!
