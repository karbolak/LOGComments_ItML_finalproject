import os
import pandas as pd

def auto_select_dataset(directory="datasets"):
    """
    Automatically selects TRAIN.csv for training and TEST.csv for testing inside the datasets folder.

    Args:
        directory (str): The directory containing datasets.

    Returns:
        tuple: (train_dataset_path, test_dataset_path)
    """
    train_path = os.path.join(directory, "TRAIN.csv")
    test_path = os.path.join(directory, "TEST.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"TRAIN.csv not found in {directory}. Ensure the dataset is available.")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"TEST.csv not found in {directory}. Ensure the dataset is available.")
    
    return train_path, test_path


def load_dataset(file_path):
    """Loads a dataset from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset file is empty: {file_path}")
    except pd.errors.ParserError:
        raise ValueError(f"Dataset file could not be parsed: {file_path}")


def select_and_load_datasets(directory):
    """Select and load training and testing datasets."""
    train_dataset_path, test_dataset_path = auto_select_dataset(directory)
    print(f"Automatically selected:\n - Training dataset: {train_dataset_path}\n - Testing dataset: {test_dataset_path}")

    train_dataset = load_dataset(train_dataset_path)
    test_dataset = load_dataset(test_dataset_path)

    target_column = train_dataset.columns[-1]
    print(f"\nAutomatically selected target feature: {target_column}")

    if train_dataset[target_column].nunique() != 2:
        raise ValueError("Target feature must be binary for Logistic Regression.")

    return train_dataset, test_dataset, target_column


def save_predictions(test_dataset, predictions, target_column, directory):
    """Save predictions to a CSV file."""
    test_dataset[target_column] = predictions
    output_file = os.path.join(directory, "predictions_with_results.csv")
    test_dataset.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
