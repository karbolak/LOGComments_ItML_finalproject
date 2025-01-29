# Introduction to Machine Learning - final project

## TODO:
- [ ] populate README.md
- [ ] add docstrings
- [ ] typehints

## How to run

```console
python main.py
```

will run the Logistic regression interface. The program does not require scikit-learn for functioning, all requirements can be found [`requirements.txt`](requirements.txt).

The folder [`/datasets/`](/datasets/) should be populated with any datasets we want to use, as the program will ask for its contents, and also the final results will be stored there. The default names of the datasets will be defined in [`data_manipulation.py`](/ml/data_manipulation.py).

The graph of the learning curves is saved in [`/plots/`](/plots/) directory. It will be overwritten with each run.

In [`hyperparameter_choice.py`](/ml/hyperparameter_choice.py) you can choose whether to go through all different hyperparameter combinations during hyperparameter tuning, or to follow the best hyperparameters we found by uncommenting the option at the end of the file (optimization is the default option):

```python
# Uncomment if you don't want to use optimization and use best hyperparameters instead
# HPARAMS = BEST_HPARAMS
```


## Structure of the codebase:
.
├── datasets
│   ├── predictions_with_results.csv
│   ├── TEST.csv
│   └── TRAIN.csv
├── main.py
├── ml
│   ├── cross_validation.py
│   ├── data_manipulation.py
│   ├── data_processing.py
│   ├── hyperparameter_choice.py
│   ├── hyperparameter_tuning.py
│   ├── learning_curve.py
│   ├── logistic_regression.py
│   ├── metric.py
│   ├── one_hot_encoder.py
│   └── trainer.py
├── plots
│   └── learning_curves_epoch_loss.png
├── README.md
└── requirements.txt


## Description of codebase:

### cross_validation.py


### data_processing.py


### learning_curve.py


### logistic_regression.py


### metric.py


### one_hot_encoder.py



## Machine learning concepts used:

### Feature normalization

### Balanced splitting of dataset

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