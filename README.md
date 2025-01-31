# Introduction to Machine Learning - Final Project

This project implements a custom Logistic Regression model with Elastic Net regularization, hyperparameter tuning, and cross-validation, without using scikit-learn. The goal is to demonstrate an end-to-end machine learning pipeline, from dataset loading to model evaluation, using pure Python and NumPy.

## How to Run

```console
python main.py
```

This will run the Logistic Regression interface. All required dependencies are listed in [`requirements.txt`](requirements.txt).

The folder [`/datasets/`](/datasets/) should contain datasets in CSV format. The program will prompt for dataset selection and store final results in this directory. The dataset file names are set in [`data_manipulation.py`](/ml/data_manipulation.py).

Learning curves will be saved in [`/plots/`](/plots/) and will be overwritten in each run.

To configure hyperparameter tuning, edit [`hyperparameter_choice.py`](/ml/hyperparameter_choice.py). By default, hyperparameter optimization is enabled. To use preselected best hyperparameters instead, uncomment:

```python
# HPARAMS = BEST_HPARAMS
```

## Codebase Structure

```python
.
├── datasets
│   ├── predictions_with_results.csv
│   ├── TEST.csv
│   └── TRAIN.csv
├── main.py
├── ml
│   ├── cross_validation.py
│   ├── data_manipulation.py
│   ├── data_processing.py
│   ├── hyperparameter_choice.py
│   ├── hyperparameter_tuning.py
│   ├── learning_curve.py
│   ├── logistic_regression.py
│   ├── metric.py
│   ├── one_hot_encoder.py
│   ├── trainer.py
├── plots
│   └── learning_curves_epoch_loss.png
├── README.md
└── requirements.txt
```

## Description of Codebase

### `cross_validation.py`
Implements k-fold stratified cross-validation to evaluate model performance across multiple subsets of data. This ensures that our model does not overfit to a specific train-test split and provides a more robust estimate of performance.

### `data_processing.py`
Handles feature preprocessing, including one-hot encoding of categorical variables and normalization of numerical features. It also detects feature types (categorical vs numerical) and ensures consistent preprocessing between training and testing datasets. Standardizing features helps improve convergence during training.

### `learning_curve.py`
Generates and saves learning curves to visualize training and validation loss over epochs. This helps in diagnosing underfitting and overfitting by observing loss trends.

### `logistic_regression.py`
Implements a custom Logistic Regression model with:
- Elastic Net regularization (L1 + L2) to improve generalization by reducing overfitting.
- Learning rate decay for efficient optimization.
- Early stopping to prevent unnecessary iterations when validation loss stops improving.
- An optimal threshold selection method to maximize classification performance.

### `metric.py`
Contains evaluation metrics, including:
- Accuracy: Measures overall correctness of predictions.
- Precision, Recall, and F1 Score: Important for imbalanced classification problems.
- ROC-AUC: Evaluates classification thresholds.
- Log Loss: Used to assess probabilistic outputs.
- Confusion Matrix: Provides a detailed breakdown of classification results.

### `one_hot_encoder.py`
Custom implementation of one-hot encoding for categorical variables. Ensures that unseen categories during inference are handled correctly, preventing errors in model predictions.

### `trainer.py`
Trains the final model using the best hyperparameters and evaluates its performance. This includes stratified splitting of the dataset to maintain class balance, training with early stopping, and plotting learning curves.

### `hyperparameter_tuning.py`
Performs exhaustive search over predefined hyperparameter grids using cross-validation. By systematically tuning parameters, we ensure that our model achieves the best possible performance without overfitting.

### `hyperparameter_choice.py`
Defines hyperparameter search space and stores best hyperparameter configurations found. This allows flexibility in selecting whether to use optimized parameters or precomputed best settings.

### `data_manipulation.py`
Loads datasets, detects target columns, and saves model predictions to CSV. Ensures that datasets are correctly formatted before training and handles errors gracefully.

### `main.py`
Executes the full machine learning pipeline: data loading, preprocessing, hyperparameter tuning, model training, evaluation, and saving predictions. Serves as the entry point for running the project.

## Machine Learning Concepts Used

### Feature Normalization
Feature scaling (Z-score normalization) is applied to numerical features to improve convergence speed and stability of gradient-based optimization. This is particularly important because Logistic Regression uses gradient-based optimization, and unnormalized data can lead to poor convergence.

### One-Hot Encoding
Categorical variables are transformed using one-hot encoding to enable logistic regression models to process them numerically. This ensures that our model can learn from categorical data without introducing ordinal relationships that do not exist.

### Elastic Net Regularization
A combination of L1 and L2 penalties is used to prevent overfitting and improve generalization. The L1 penalty helps with feature selection by forcing some coefficients to zero, while the L2 penalty helps with stability by reducing extreme weight values.

### Learning Rate Decay
Gradual reduction of the learning rate helps the model converge efficiently and avoid overshooting the optimal solution. This is implemented as different decay strategies (time-based, exponential, and step decay) to allow flexibility in tuning.

### Early Stopping
Monitoring validation loss prevents unnecessary iterations and overfitting by stopping training when no improvement is observed. This helps in reducing computation costs and improving model generalization.

### Hyperparameter Tuning
Cross-validation is used to optimize hyperparameters such as:
- Learning rate: Determines the step size for weight updates.
- Number of iterations: Ensures sufficient training while avoiding excessive computation.
- Regularization strength (L1/L2 ratio): Controls the trade-off between feature selection and weight penalization.
- Learning rate decay type and rate: Ensures adaptive learning over training iterations.

### Balanced Splitting of Dataset
Stratified splitting ensures class distribution remains consistent across training, validation, and test sets. This is especially important for imbalanced datasets to prevent bias in model training.

### Cross-Validation
k-fold cross-validation provides a robust estimate of model performance and helps prevent overfitting to specific dataset splits. This ensures that our chosen hyperparameters generalize well across unseen data.

## Motivation
This project was developed to demonstrate a fully functional machine learning pipeline without reliance on external libraries like scikit-learn. The custom implementation provides fine-grained control over the model training process and enhances understanding of logistic regression internals. By implementing each component from scratch, we gain deeper insights into how different ML concepts interact and how design choices impact model performance.

---

Cross-Validation Summary:
Accuracy: Mean = 0.8181, Std = 0.0205
Loss: Mean = 0.3926, Std = 0.0135

Best Hyperparameters:
n_iterations: 1000
learning_rate: 0.1
regularization: 0.0
decay_rate: 0.01
decay_type: none



### Using Class-Weighted Loss to Penalize Misclassified `bot` Cases

Incorporating a class-weighted loss helps handle imbalanced datasets by assigning higher penalties to misclassified samples in underrepresented classes.

#### Implementation Steps:

1. **Add Class Weights**:
   - Modify the `fit` method to include weights for each class:
     - Weight for `bot` (positive class) should be higher to penalize false negatives.
     - Weight for `non-bot` (negative class) can remain lower.

```python
def fit(self, X: np.ndarray, y: np.ndarray, class_weights=None):
    if y.ndim > 1 and y.shape[1] == 2:
        y = y[:, 1]  # Convert one-hot to single-class labels if necessary

    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    # Assign default weights if none provided
    if class_weights is None:
        class_weights = {0: 1.0, 1: 1.0}

    for _ in range(self.n_iterations):
        model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(model)

        # Compute weighted gradients
        weights_vector = np.array([class_weights[int(label)] for label in y])
        errors = predictions - y

        dw = (1 / n_samples) * np.dot(X.T, weights_vector * errors) + self.regularization * self.weights
        db = (1 / n_samples) * np.sum(weights_vector * errors)

        # Update weights and bias
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    self.trained = True
```

2. **Determine Class Weights**:
   - Use the class distribution in the dataset to calculate weights:
     - Example: `class_weights = {0: 1.0, 1: len(y) / np.sum(y)}` (inverse frequency).

3. **Fit the Model with Weights**:
   - Call `fit` with the `class_weights` hyperparameter.

---

### Combining Threshold Adjustment and Weighted Loss

For the best results:
1. **Train with Weighted Loss**: 
   - Ensure that the model prioritizes reducing errors on underrepresented classes.
2. **Optimize the Threshold**:
   - After training, vary the decision threshold to find the best trade-off between precision and recall.

---

### Example of Combined Implementation in `main.py`

```python
# Training with class weights
class_weights = {0: 1.0, 1: len(y_train) / np.sum(y_train)}
final_model = LogisticRegression(
    learning_rate=best_hparams['learning_rate'],
    n_iterations=best_hparams['n_iterations'],
    regularization=best_hparams['regularization']
)
final_model.fit(X_train, y_train, class_weights=class_weights)

# Optimize threshold
thresholds = [0.5, 0.4, 0.3, 0.2]
for threshold in thresholds:
    predictions = final_model.predict(X_pred, threshold=threshold)
    recall = Recall()(original_target, predictions)
    precision = Precision()(original_target, predictions)
    print(f"Threshold: {threshold} - Recall: {recall:.4f}, Precision: {precision:.4f}")
```


The collection of best hyperparameters so far:

Best Hyperparameters:
n_iterations: 5000
learning_rate: 0.1
regularization: 0.0001
decay_rate: 0.006
decay_type: time

Model Performance Metrics:
Accuracy: 0.7511
Precision: 0.7624
Recall: 0.6509
F1 Score: 0.7023
ROC-AUC: 0.8833
Log Loss: 0.4967
Confusion Matrix: 
[[215  43]
 [ 74 138]]

Best Hyperparameters:
n_iterations: 2000
learning_rate: 0.1
regularization: 0.0001
decay_rate: 0.01
decay_type: time

Model Performance Metrics:
Accuracy: 0.7830
Precision: 0.7037
Recall: 0.8962
F1 Score: 0.7884
ROC-AUC: 0.8285
Log Loss: 0.4945
Confusion Matrix: 
[[178  80]
 [ 22 190]]

Best Hyperparameters:
n_iterations: 5000
learning_rate: 0.05
regularization: 0.001
l1_ratio: 0.2
decay_rate: 0.006
decay_type: time

Model Performance Metrics:
Accuracy: 0.7809
Precision: 0.7026
Recall: 0.8915
F1 Score: 0.7859
ROC-AUC: 0.8275
Log Loss: 0.4948
Confusion Matrix: 
[['TN:178' 'FP:80']
 ['FN:23' 'TP:189']]