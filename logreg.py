import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml.functional_feature import detect_feature_types
from ml.logistic_regression import LogisticRegression
from ml.metric import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, ROCAUC, LogLoss
from ml.one_hot_encoder import OneHotEncoder

class CrossValidator:
    def __init__(self, 
                 model_class=LogisticRegression, 
                 n_splits=5, 
                 learning_rate=0.01, 
                 n_iterations=1000):
        self.n_splits = n_splits
        self.model_class = model_class
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
    
    def _stratified_split(self, X, y):
        """Perform stratified k-fold cross-validation split."""
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        
        pos_folds = np.array_split(pos_indices, self.n_splits)
        neg_folds = np.array_split(neg_indices, self.n_splits)
        
        splits = []
        for i in range(self.n_splits):
            val_pos = pos_folds[i]
            val_neg = neg_folds[i]
            
            val_indices = np.concatenate([val_pos, val_neg])
            train_indices = np.setdiff1d(np.arange(len(y)), val_indices)
            
            splits.append((train_indices, val_indices))
        
        return splits
    
    def cross_validate(self, X, y):
        """Perform cross-validation with detailed metrics."""
        splits = self._stratified_split(X, y)
        
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        accuracy_calc = Accuracy()
        precision_calc = Precision()
        recall_calc = Recall()
        f1_calc = F1Score()
        
        for train_idx, val_idx in splits:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self.model_class(
                learning_rate=self.learning_rate, 
                n_iterations=self.n_iterations
            )
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_val)
            
            metrics['accuracy'].append(accuracy_calc(y_val, predictions))
            metrics['precision'].append(precision_calc(y_val, predictions))
            metrics['recall'].append(recall_calc(y_val, predictions))
            metrics['f1_score'].append(f1_calc(y_val, predictions))
        
        # Compute summary statistics
        summary = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            } for metric, values in metrics.items()
        }
        
        return metrics, summary
    
class LearningCurve:
    def __init__(self, 
                 model_class=LogisticRegression, 
                 learning_rate=0.01, 
                 n_iterations=1000,
                 metric='accuracy'):
        self.model_class = model_class
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.metric = metric
        self.metric_calculator = {
            'accuracy': Accuracy(),
            'precision': Precision(),
            'recall': Recall(),
            'f1_score': F1Score()
        }.get(metric.lower(), Accuracy())

    def generate(self, X, y, train_sizes=None):
        """
        Generate learning curves for the model.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            train_sizes (list): Proportions or absolute numbers of training samples
        
        Returns:
            tuple: train_sizes, train_scores, validation_scores
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        train_scores, val_scores = [], []

        for size in train_sizes:
            # Determine the number of samples to use
            if size <= 1.0:
                n_train_samples = int(len(X) * size)
            else:
                n_train_samples = int(size)

            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Split data
            X_train = X_shuffled[:n_train_samples]
            y_train = y_shuffled[:n_train_samples]
            X_val = X_shuffled[n_train_samples:]
            y_val = y_shuffled[n_train_samples:]

            # Train model
            model = self.model_class(
                learning_rate=self.learning_rate, 
                n_iterations=self.n_iterations
            )
            model.fit(X_train, y_train)

            # Compute training and validation scores
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_score = self.metric_calculator(y_train, train_pred)
            val_score = self.metric_calculator(y_val, val_pred)

            train_scores.append(train_score)
            val_scores.append(val_score)

        return train_sizes, train_scores, val_scores

def plot_learning_curves(train_sizes, train_scores, val_scores, metric='Accuracy'):
    """
    Plot learning curves.
    
    Args:
        train_sizes (list): Training sample sizes
        train_scores (list): Training scores
        val_scores (list): Validation scores
        metric (str): Metric name for y-axis label
    """
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, label='Training Score', color='blue', marker='o')
    plt.plot(train_sizes, val_scores, label='Validation Score', color='red', marker='o')
    
    plt.title(f'Learning Curves ({metric})')
    plt.xlabel('Training Sample Size')
    plt.ylabel(f'{metric} Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./plots/learning_curves_{metric}.png')
    plt.close()

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

def plot_cross_validation_results(metrics):
    """Visualize cross-validation results."""
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 6))
    for metric, values in metrics.items():
        plt.plot(values, label=metric)
    
    plt.title('Cross-Validation Performance Metrics')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/cross_validation_results.png')
    plt.close()

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
    cross_validator = CrossValidator(n_splits=5)
    cv_metrics, cv_summary = cross_validator.cross_validate(X_train, y_train)

    # Print Cross-Validation Results
    print("\nCross-Validation Summary:")
    for metric, stats in cv_summary.items():
        print(f"{metric.capitalize()}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")

    # Visualize Cross-Validation Results
    plot_cross_validation_results(cv_metrics)
    print("Cross-validation results plot saved to 'cross_validation_results.png'")

    # Train final model
    print("\nTraining final Logistic Regression model...")
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    
    learning_curve = LearningCurve(n_iterations=1000)
    
    all_metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    metrics_to_plot = ['f1_score']
    
    for metric in metrics_to_plot:
        learning_curve.metric = metric
        train_sizes, train_scores, val_scores = learning_curve.generate(X_train, y_train)
        plot_learning_curves(train_sizes, train_scores, val_scores, metric.capitalize())
        print(f"Learning curves for {metric} saved to 'learning_curves_{metric}.png'")

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
    predictions = model.predict(X_pred)

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