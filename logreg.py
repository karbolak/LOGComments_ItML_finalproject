import os
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from ml.functional_feature import detect_feature_types
from ml.logistic_regression import LogisticRegression

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

def preprocess_and_transform(dataset, input_features, train_artifacts, dataset_type="training"):
    """Preprocess and transform features using artifacts."""
    print(f"\nPreprocessing {dataset_type} dataset...")
    processed_frames = []

    for feature in input_features:
        name = feature.name
        if dataset_type == "training":
            # Create and fit encoder during training
            encoder = OneHotEncoder(categories="auto", sparse_output=True, handle_unknown="ignore")
            encoder.fit(dataset[[name]])
            train_artifacts[name] = {"type": "OneHotEncoder", "encoder": encoder}
            data = encoder.transform(dataset[[name]]).toarray()
        else:
            # Use the fitted encoder from artifacts during prediction
            artifact = train_artifacts.get(name)
            if artifact is None:
                print(f"Warning: No artifact found for feature '{name}'. Skipping this feature.")
                continue
            if artifact["type"] == "OneHotEncoder":
                encoder = artifact["encoder"]
                data = encoder.transform(dataset[[name]]).toarray()
            else:
                print(f"Unsupported artifact type for feature '{name}'. Skipping this feature.")
                continue

        # Handle expanded columns (e.g., one-hot encoding)
        expanded_columns = [f"{name}_{i}" for i in range(data.shape[1])]
        processed_frames.append(pd.DataFrame(data, columns=expanded_columns))

    if not processed_frames:
        raise ValueError(f"No features were processed for the {dataset_type} dataset. Check feature alignment.")

    return pd.concat(processed_frames, axis=1).values

def run_pipeline(directory="datasets"):
    """Runs an interactive pipeline for Logistic Regression."""
    # Select and load training dataset
    print("Select the training dataset:")
    train_dataset_path = select_dataset(directory)
    train_dataset = load_dataset(train_dataset_path)
    print(f"Loaded training dataset: {train_dataset_path}")

    # Detect features
    print("Detecting feature types...")
    features = detect_feature_types(train_dataset)
    feature_names = [f.name for f in features]

    # Select target feature
    print("\nAvailable Features:")
    for i, name in enumerate(feature_names):
        print(f"{i + 1}. {name}")

    target_index = int(input("\nSelect the target feature by number: ")) - 1
    if target_index < 0 or target_index >= len(feature_names):
        raise ValueError("Invalid choice. Please select a valid feature.")

    target_column = feature_names[target_index]
    input_features = [f for f in features if f.name != target_column]

    # Ensure target feature is binary
    if train_dataset[target_column].nunique() != 2:
        raise ValueError("Target feature must be binary for Logistic Regression.")

    # Preprocess training dataset
    print("\nPreprocessing training dataset...")
    train_artifacts = {}
    X_train = preprocess_and_transform(train_dataset, input_features, train_artifacts, dataset_type="training")
    y_train = train_dataset[target_column].values

    # Save training artifacts
    with open("train_artifacts.pkl", "wb") as f:
        pickle.dump(train_artifacts, f)

    # Train the model
    print("Training Logistic Regression model...")
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)

    # Save the trained model
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Select and load prediction dataset
    print("\nSelect the prediction dataset:")
    prediction_dataset_path = select_dataset(directory)
    prediction_dataset = load_dataset(prediction_dataset_path)
    print(f"Loaded prediction dataset: {prediction_dataset_path}")

    # Load training artifacts
    with open("train_artifacts.pkl", "rb") as f:
        train_artifacts = pickle.load(f)

    # Ensure target column is excluded from input features
    if target_column in prediction_dataset.columns:
        original_target = prediction_dataset[target_column].values
        prediction_dataset = prediction_dataset.drop(columns=[target_column])
    else:
        raise ValueError("Target column missing in the prediction dataset.")

    # Preprocess prediction dataset
    X_pred = preprocess_and_transform(prediction_dataset, input_features, train_artifacts, dataset_type="prediction")

    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(X_pred)

    # Calculate metrics
    print("Calculating metrics...")
    accuracy = (predictions == original_target).mean()
    precision = precision_score(original_target, predictions)
    recall = recall_score(original_target, predictions)
    f1 = f1_score(original_target, predictions)
    roc_auc = roc_auc_score(original_target, predictions)
    conf_matrix = confusion_matrix(original_target, predictions)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Save predictions
    prediction_dataset[target_column] = predictions
    output_file = os.path.join(directory, "predictions_with_results.csv")
    prediction_dataset.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    # Default datasets directory
    datasets_directory = "datasets"

    # Ensure the datasets directory exists
    if not os.path.exists(datasets_directory):
        os.makedirs(datasets_directory)
        print(f"Created directory for datasets: {datasets_directory}")

    # Run the pipeline
    run_pipeline(directory=datasets_directory)