# Model Loading Examples

This directory contains examples demonstrating how to use the flexible model loading functionality provided by the `model_loader.py` module.

## Overview

The `model_loader.py` module was created to address the issue of hardcoded model paths in the codebase. Previously, the code was bound to specific model files (e.g., `"logreg_Fits_Topic_Code_model.pkl"`), which made it inflexible and prone to errors when the specified model file didn't exist.

The new implementation provides:

1. **Flexible model loading**: Models can be loaded by specifying the model type and target column, or by providing a direct path to the model file.
2. **Error handling**: When a model cannot be loaded (e.g., because the file doesn't exist), an informative message is displayed instead of raising an exception.
3. **Unified interface**: The same interface can be used to load both logistic regression models (.pkl files) and neural network models (.pt files).

## Example Script

The `model_loading_example.py` script demonstrates how to use the `model_loader.py` module in various scenarios:

1. Loading a logistic regression model by type and target
2. Loading a neural network model by type and target
3. Loading a model with a direct path
4. Trying to load any existing model in the models directory

Each example shows how to handle both successful and unsuccessful model loading scenarios, with appropriate information messages displayed to the user.

## Usage

To run the example script:

```bash
python examples/model_loading_example.py
```

## Integration with Existing Code

To use the flexible model loading functionality in your own code, replace hardcoded model loading code like:

```python
model_path = os.path.join("models", "logreg_Fits_Topic_Code_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)
```

with the new flexible approach:

```python
from model_loader import load_model_with_info

# Option 1: Load by model type and target column
result = load_model_with_info(model_type="logreg", target_column="Fits_Topic_Code")
if result['success']:
    model = result['model']
    # Use the model for predictions
else:
    print(f"Info: {result['message']}")
    # Handle the case where the model couldn't be loaded

# Option 2: Load by direct path
result = load_model_with_info(model_path="models/custom_model.pkl")
if result['success']:
    model = result['model']
    # Use the model for predictions
else:
    print(f"Info: {result['message']}")
    # Handle the case where the model couldn't be loaded
```

This approach ensures that your code is not bound to a specific model file and provides informative messages when a model cannot be loaded.