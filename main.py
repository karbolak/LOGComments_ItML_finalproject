import os
import pandas as pd
import numpy as np
from ml.cross_validation import CrossValidator
from ml.learning_curve import plot_learning_curves
from ml.functional_feature import detect_feature_types
from ml.logistic_regression import LogisticRegression
from ml.metric import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, ROCAUC, LogLoss
from ml.one_hot_encoder import OneHotEncoder

def load_dataset(file_path):
    """Loads a dataset from a CSV file."""
    return pd.read_csv(file_path)

def select_dataset(directory="datasets"):
    """Prompts the user to select a dataset from a list."""
    datasets = [f for f in os.listdir(directory) if f.endswith(".csv")]
    if not datasets:
        raise ValueError(f"No CSV datasets found in '{directory}' directory.")
    
    print("\nAvailable Datasets:")
    for i, dataset in enumerate(datasets):
        print(f"{i + 1}. {dataset}")
    
    choice = int(input("\nSelect a dataset by number: ")) - 1
    if choice < 0 or choice >= len(datasets):
        raise ValueError("Invalid choice. Please select a valid dataset.")
    
    return os.path.join(directory, datasets[choice])

def train_validation_split(X, y, val_size=0.2, random_seed=42):
    """
    Splits the dataset into training and validation sets.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        val_size (float): Proportion of the data to use as validation (0 < val_size < 1).
        random_seed (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    np.random.seed(random_seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    split_idx = int(len(indices) * (1 - val_size))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]

    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    return X_train, X_val, y_train, y_val


def preprocess_and_transform(dataset, input_features, train_artifacts, dataset_type="training"):
    """Preprocess and transform features using artifacts."""
    print(f"\nPreprocessing {dataset_type} dataset...")
    processed_frames = []

    for feature in input_features:
        name = feature.name
        column_data = dataset[[name]].values

        if dataset_type == "training":
            encoder = OneHotEncoder()
            encoder.fit(column_data)
            train_artifacts[name] = {"type": "OneHotEncoder", "encoder": encoder}
            data = encoder.transform(column_data)
        else:
            artifact = train_artifacts.get(name)
            if artifact is None:
                print(f"Warning: No artifact found for feature '{name}'. Skipping this feature.")
                continue
            if artifact["type"] == "OneHotEncoder":
                encoder = artifact["encoder"]
                data = encoder.transform(column_data)
            else:
                print(f"Unsupported artifact type for feature '{name}'. Skipping this feature.")
                continue

        expanded_columns = [f"{name}_{i}" for i in range(data.shape[1])]
        processed_frames.append(pd.DataFrame(data, columns=expanded_columns))

    if not processed_frames:
        raise ValueError(f"No features were processed for the {dataset_type} dataset. Check feature alignment.")

    return pd.concat(processed_frames, axis=1).values

def run_pipeline(directory="datasets", artifacts_dir="artifacts"):
    """Runs an interactive pipeline for Logistic Regression."""
    os.makedirs(artifacts_dir, exist_ok=True)

    # Select and load training dataset
    print("Select the training dataset:")
    train_dataset_path = select_dataset(directory)
    train_dataset = load_dataset(train_dataset_path)
    print(f"Loaded training dataset: {train_dataset_path}")

    # Detect and select features
    features = detect_feature_types(train_dataset)
    feature_names = [f.name for f in features]

    print("\nAvailable Features:")
    for i, name in enumerate(feature_names):
        print(f"{i + 1}. {name}")

    target_index = int(input("\nSelect the target feature by number: ")) - 1
    target_column = feature_names[target_index]
    input_features = [f for f in features if f.name != target_column]

    # Validate target feature
    if train_dataset[target_column].nunique() != 2:
        raise ValueError("Target feature must be binary for Logistic Regression.")

    # Preprocess training dataset
    train_artifacts = {}
    X_train = preprocess_and_transform(train_dataset, input_features, train_artifacts, dataset_type="training")
    y_train = train_dataset[target_column].values

    # Cross-Validation
    cross_validator = cross_validator = CrossValidator()

    best_model, best_params, cv_summary = cross_validator.cross_validate(X_train, y_train)

    # Print Cross-Validation Results
    print("\nCross-Validation Summary:")
    for metric, stats in cv_summary.items():
        print(f"{metric.capitalize()}: Mean = {stats['mean']:.4f}, Std = {stats['std']:.4f}")
    
    print("\nBest Parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # Use best parameters for the final model
    print("\nTraining final Logistic Regression model with best parameters...")
    # Split the dataset for validation during final training
    X_train, X_val, y_train, y_val = train_validation_split(X_train, y_train, val_size=0.2)

    # Train the final model
    final_model = LogisticRegression(
        learning_rate=best_params['learning_rate'],
        n_iterations=best_params['n_iterations'],
        regularization=best_params['regularization'],
        epochs=10,
        decay_type=best_params['decay_type'],
        decay_rate=best_params['decay_rate']
    )
    final_model.fit(X_train, y_train, validation_data=(X_val, y_val))
    print("Final model training complete.")
    
    plot_learning_curves(final_model.train_losses, final_model.val_losses)

    
    # Prediction pipeline (rest of the existing code)
    print("\nSelect the prediction dataset:")
    prediction_dataset_path = select_dataset(directory)
    prediction_dataset = load_dataset(prediction_dataset_path)

    if target_column in prediction_dataset.columns:
        original_target = prediction_dataset[target_column].values
        prediction_dataset = prediction_dataset.drop(columns=[target_column])
    else:
        raise ValueError("Target column missing in the prediction dataset.")

    X_pred = preprocess_and_transform(prediction_dataset, input_features, train_artifacts, dataset_type="prediction")
    predictions = final_model.predict(X_pred)

    # Metrics calculation
    metrics_calculators = {
        'Accuracy': Accuracy(),
        'Precision': Precision(),
        'Recall': Recall(),
        'F1 Score': F1Score(),
        'ROC-AUC': ROCAUC(),
        'Log Loss': LogLoss(),
        'Confustion Matrix': ConfusionMatrix(),
    }

    print("\nModel Performance Metrics:")
    for name, calculator in metrics_calculators.items():
        score = calculator(original_target, predictions)
        if name == 'Confustion Matrix':
            print(f"{name}: \n{score}")
        else:
            print(f"{name}: {score:.4f}")


    # Save predictions
    prediction_dataset[target_column] = predictions
    output_file = os.path.join(directory, "predictions_with_results.csv")
    prediction_dataset.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")

if __name__ == "__main__":
    datasets_directory = "datasets"
    artifacts_directory = "artifacts"

    os.makedirs(datasets_directory, exist_ok=True)
    run_pipeline(directory=datasets_directory, artifacts_dir=artifacts_directory)
