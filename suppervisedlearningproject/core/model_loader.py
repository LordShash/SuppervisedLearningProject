import os
import joblib
import torch
from typing import Any, Optional, Union, Dict

# Importiere die Konfiguration und richte das Logging ein
# Angepasst für die neue Paketstruktur
from suppervisedlearningproject.utils import setup_logging, MODELS_DIR, CHECKPOINTS_DIR

# Konfiguration des Loggings mit dem zentralen Setup
logger = setup_logging(__name__)

def load_model(model_type: str = None, target_column: str = None, model_path: str = None) -> Optional[Any]:
    """
    Loads a trained model from disk with flexible path options.

    Args:
        model_type (str, optional): Type of model to load ('logreg' or 'nn'). 
                                   Required if model_path is not provided.
        target_column (str, optional): Target column used for training the model.
                                      Required if model_path is not provided.
        model_path (str, optional): Direct path to the model file. If provided, 
                                   model_type and target_column are ignored.

    Returns:
        Optional[Any]: The loaded model or None if the model could not be loaded.
    """
    try:
        # If model_path is not provided, construct it from model_type and target_column
        if not model_path:
            if not model_type or not target_column:
                logger.warning("Either model_path or both model_type and target_column must be provided.")
                return None

            # Construct model path based on model type and target column
            if model_type == "logreg":
                model_path = os.path.join(MODELS_DIR, f"logreg_{target_column}_model.pkl")
            elif model_type == "nn":
                model_path = os.path.join(CHECKPOINTS_DIR, f"{target_column}_best_model.pt")
            else:
                logger.warning(f"Unsupported model type: {model_type}. Supported types are 'logreg' and 'nn'.")
                return None

        # Check if model file exists
        if not os.path.exists(model_path):
            logger.info(f"Model file not found: {model_path}")
            return None

        # Load the model based on file extension
        if model_path.endswith('.pkl'):
            logger.info(f"Loading logistic regression model from: {model_path}")
            return joblib.load(model_path)
        elif model_path.endswith('.pt'):
            logger.info(f"Loading neural network model from: {model_path}")
            # For neural network models, we need to know the architecture
            # This is a simplified example - in practice, you might need to load metadata first
            # to determine the correct architecture
            model_data = torch.load(model_path)

            # If the model was saved with metadata (as a dictionary)
            if isinstance(model_data, dict) and 'model_state_dict' in model_data:
                # Here you would reconstruct the model architecture and load the state dict
                # This is a placeholder for the actual implementation
                logger.info("Model contains metadata. Loading state dict...")
                # model = create_model_from_metadata(model_data['metadata'])
                # model.load_state_dict(model_data['model_state_dict'])
                return model_data
            else:
                # If the model was saved as just the state dict
                logger.info("Model saved as state dict only.")
                # You would need to know the architecture in advance
                # model = YourModelClass(...)
                # model.load_state_dict(model_data)
                return model_data
        else:
            logger.warning(f"Unsupported model file format: {model_path}")
            return None

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def load_model_with_info(model_type: str = None, target_column: str = None, model_path: str = None) -> Dict[str, Any]:
    """
    Loads a model and returns both the model and information about the loading process.

    Args:
        Same as load_model function.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'model': The loaded model or None if loading failed
            - 'success': Boolean indicating if loading was successful
            - 'message': Information message about the loading process
    """
    result = {
        'model': None,
        'success': False,
        'message': ""
    }

    # Try to load the model
    model = load_model(model_type, target_column, model_path)

    if model is not None:
        result['model'] = model
        result['success'] = True

        if model_path:
            result['message'] = f"Model successfully loaded from: {model_path}"
        else:
            result['message'] = f"Model successfully loaded: {model_type}_{target_column}"
    else:
        # Model loading failed
        if not model_path and (not model_type or not target_column):
            result['message'] = "Model loading failed: Either model_path or both model_type and target_column must be provided."
        elif model_path and not os.path.exists(model_path):
            result['message'] = f"Model loading failed: File not found at {model_path}"
        else:
            result['message'] = "Model loading failed. Check logs for details."

    return result

# Example usage
if __name__ == "__main__":
    # Example 1: Load a logistic regression model by type and target
    result = load_model_with_info(model_type="logreg", target_column="Fits_Topic_Code")
    if result['success']:
        print(f"Success: {result['message']}")
        # Use the model for predictions
    else:
        print(f"Error: {result['message']}")

    # Example 2: Load a model with direct path
    result = load_model_with_info(model_path=os.path.join(MODELS_DIR, "custom_model.pkl"))
    if result['success']:
        print(f"Success: {result['message']}")
        # Use the model for predictions
    else:
        print(f"Error: {result['message']}")