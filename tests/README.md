# Testing Guide for the Text Classification Application

This directory contains tests for the Text Classification Application. This guide will help you understand how to run existing tests and write new ones.

## Running Tests

You can run all tests using pytest:

```bash
# From the project root directory
pytest tests/

# With coverage report
pytest tests/ --cov=src
```

## Test Structure

The tests are organized to mirror the structure of the source code:

- `test_main.py`: Tests for the main functionality
- `test_data_loader.py`: Tests for the data loading functionality
- `test_model_loader.py`: Tests for the model loading functionality
- `test_train_logreg.py`: Tests for the logistic regression training
- `test_train_nn.py`: Tests for the neural network training

## Writing New Tests

When writing new tests, follow these guidelines:

1. Create a new test file if testing a new module, following the naming convention `test_<module_name>.py`
2. Use pytest fixtures for common setup and teardown
3. Mock external dependencies when appropriate
4. Test both success and failure cases
5. Include docstrings that explain what each test is checking

### Example Test

```python
import pytest
from src.data_loader import load_data

def test_load_data_with_valid_target():
    """Test that load_data works with a valid target column."""
    X, y = load_data(target_column="Fits_Topic_Code", max_features=10)
    assert X is not None
    assert y is not None
    assert X.shape[1] == 10  # Should have 10 features

def test_load_data_with_invalid_target():
    """Test that load_data raises an error with an invalid target column."""
    with pytest.raises(KeyError):
        load_data(target_column="NonExistentColumn")
```

## Test Coverage

Aim for high test coverage, especially for critical components:

- Data loading and preprocessing
- Model training and evaluation
- Error handling
- Configuration management

## Continuous Integration

Tests are automatically run in the CI pipeline when changes are pushed to the repository. Make sure all tests pass before submitting a pull request.