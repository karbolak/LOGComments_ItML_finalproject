# LOGComments_ItML_finalproject

Base is oop project at an early stage, most of it is useless (see the whole app subfolder) but I did not want to bother with removing it and fixing the errors, that can be done later.

```console
python main.py
```

will run the Logistic regression interface I wrote that allows the basic functioning needed for our purposes.

The folder **datasets** should be populated with any datasets we want to use, as the program will ask for its contents, and also the final results will be stored there.

TODO:
- [x] Validation / Cross-Validation
- [x] detect overfitting / underfitting
- [x] Metrics
- [x] Codebase cleanup
- [x] one-hot encoding not by sklearn
- [x] regularisation to combat overfitting
- [x] introduced epochs
- [ ] figure out artifact situation
- [ ] add percentage progress during training
- [x] check what hyperparameter values are selected (print)
- [ ] check if changing accuracy to F1 will improve the results
- [x] introduce learning rate decay
- [ ] why constant loss over epochs
- [x] check different epoch numbers for main model and CV
- [ ] check out Platt Scaling
- [ ] change the LogReg threshold (see end of the file)
- [ ] adjust possible hyperparameter values based on logarithmic ranges around the best values
- [ ] figure out the n_iterations (gradient) adjustment
- [x] early stopping

Cross-Validation Insights:

    The best results were achieved with higher learning rates (0.1) and no regularization (0.0), suggesting that over-regularization harms model performance.
    Additional tuning of n_iterations or introducing learning rate decay could further improve performance.

Learning Curve Issue:

    The constant loss values across epochs suggest:
        Validation set may overlap with the training set.
        Lack of proper splitting in the LearningCurve class or insufficient samples for validation.
    Solution: Ensure the validation set is strictly separated and contains sufficient samples.

Log Loss Concern:

    High log loss (5.5675) during prediction indicates poorly calibrated probabilities.
    Solution: Use probability calibration methods (e.g., Platt Scaling) before converting probabilities to binary predictions.

Model Strengths and Weaknesses:

    Strengths: High accuracy and precision indicate the model is effective at identifying non-bots.
    Weaknesses: Moderate recall and high false negatives suggest the model struggles with detecting all bots.
    Solution: Adjust the classification threshold to improve recall, or use class-weighted loss to penalize misclassified bot cases.




Results from the first run:
Accuracy: 0.77
Precision: 0.75
Recall: 0.78
F1 Score: 0.76
ROC-AUC: 0.77
Log Loss: 7.84
Confusion Matrix:
[[276  82]
 [ 70 242]]

Results after deuselessification:
Accuracy: 0.77
Precision: 0.75
Recall: 0.77
F1 Score: 0.76
ROC-AUC: 0.77
Log Loss: 7.89
Confusion Matrix:
[[276  82]
 [ 71 241]]

After first regularization:
Model Performance Metrics:
Accuracy: 0.8239
Precision: 0.8070
Recall: 0.8173
F1 Score: 0.8121
ROC-AUC: 0.8122
Log Loss: 6.0830
Confustion Matrix: 
[[297  61]
 [ 57 255]]

After introducing epochs:
Model Performance Metrics:
Accuracy: 0.8388
Precision: 0.8517
Recall: 0.7917
F1 Score: 0.8206
ROC-AUC: 0.8306
Log Loss: 5.5675
Confustion Matrix: 
[[315  43]
 [ 65 247]]

W/ Learning decay:
Model Performance Metrics:
Accuracy: 0.8119
Precision: 0.7925
Recall: 0.8077
F1 Score: 0.8000
ROC-AUC: 0.8052
Log Loss: 6.4954
Confustion Matrix: 
[[292  66]
 [ 60 252]]

Model Performance Metrics:
Accuracy: 0.8433
Precision: 0.8657
Recall: 0.7853
F1 Score: 0.8235
ROC-AUC: 0.8413
Log Loss: 5.4128
Confustion Matrix: 
[[320  38]
 [ 67 245]]


---------
The percentage of all correct predictions (TP + TN) out of the total predictions
---------
Precision -> Out of all predicted positives, x% were actually positive.
---------
Recall -> Out of all actual positives, x% were correctly identified.
---------
F1 Score -> The harmonic mean of precision and recall.
---------
ROC-AUC -> Measures the ability of your model to separate classes. A score of 0.5 means random guessing, and closer to 1 is better.
---------
Log Loss -> The penalty for how far off your predicted probabilities are from the true labels. 0.1 - 1 is normal
---------
Confusion Matrix:
    confusion_matrix
    [TP   FP]"
    [FP   FN]
---------


Cross-Validation Summary:
Accuracy: Mean = 0.8181, Std = 0.0205
Loss: Mean = 0.3926, Std = 0.0135

Best Hyperparameters:
n_iterations: 1000
learning_rate: 0.1
regularization: 0.0
decay_rate: 0.01
decay_type: none

### Adjusting the Classification Threshold to Improve Recall

The default threshold for logistic regression is 0.5, meaning predictions with probabilities ≥ 0.5 are classified as positive (e.g., bot). Lowering this threshold increases recall at the cost of precision, as more cases are classified as positive.


2. **Set a Lower Threshold**:
   - Test thresholds such as `0.4`, `0.3`, etc., to find an optimal balance between recall and precision.

3. **Evaluate the Model**:
   - Use metrics like precision, recall, F1 score, and confusion matrix for each threshold to decide on the best trade-off.

---

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

---

Let me know if you'd like help implementing this or analyzing the results!

