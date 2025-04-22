import os
import sys
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Import the model loader
from model_loader import load_model_with_info

def main():
    print("Model Loading Example")
    print("=" * 50)
    
    # Example 1: Try to load a logistic regression model by type and target
    print("\nExample 1: Loading a logistic regression model by type and target")
    result = load_model_with_info(model_type="logreg", target_column="Fits_Topic_Code")
    if result['success']:
        print(f"Success: {result['message']}")
        print("Model loaded successfully. You can now use it for predictions.")
    else:
        print(f"Info: {result['message']}")
        print("You need to train a model first or specify a different model.")
    
    # Example 2: Try to load a neural network model by type and target
    print("\nExample 2: Loading a neural network model by type and target")
    result = load_model_with_info(model_type="nn", target_column="Fits_Topic_Code")
    if result['success']:
        print(f"Success: {result['message']}")
        print("Model loaded successfully. You can now use it for predictions.")
    else:
        print(f"Info: {result['message']}")
        print("You need to train a model first or specify a different model.")
    
    # Example 3: Try to load a model with direct path
    print("\nExample 3: Loading a model with direct path")
    model_path = os.path.join("models", "custom_model.pkl")
    result = load_model_with_info(model_path=model_path)
    if result['success']:
        print(f"Success: {result['message']}")
        print("Model loaded successfully. You can now use it for predictions.")
    else:
        print(f"Info: {result['message']}")
        print(f"The model file {model_path} does not exist.")
    
    # Example 4: Try to load a model that exists (if any model exists in the models directory)
    print("\nExample 4: Trying to load any existing model in the models directory")
    models_dir = "models"
    if os.path.exists(models_dir) and os.listdir(models_dir):
        # Get the first model file in the directory
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') or f.endswith('.pt')]
        if model_files:
            model_path = os.path.join(models_dir, model_files[0])
            print(f"Found model file: {model_path}")
            result = load_model_with_info(model_path=model_path)
            if result['success']:
                print(f"Success: {result['message']}")
                print("Model loaded successfully. You can now use it for predictions.")
            else:
                print(f"Info: {result['message']}")
        else:
            print("No model files found in the models directory.")
    else:
        print("Models directory does not exist or is empty.")
    
    print("\nDemonstration of flexible model loading completed.")

if __name__ == "__main__":
    main()