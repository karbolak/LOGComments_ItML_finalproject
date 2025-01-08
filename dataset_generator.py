import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # noqa: F401

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples and features
n_samples = 1000
n_features = 5

# Generate features for training
X = pd.DataFrame({
    "feature_1": np.random.normal(50, 10, n_samples),  # Normally distr
    "feature_2": np.random.uniform(0, 100, n_samples),  # Uniformly distr
    "feature_3": np.random.binomial(100, 0.5, n_samples),  # Binomial distr
    "feature_4": np.random.poisson(5, n_samples),  # Poisson distr
    "feature_5": np.random.exponential(1, n_samples)  # Exponential distr
})

# Generate a target variable that depends on some of the features
# Using a linear combination plus some random noise
y = (2 * X["feature_1"] + 3 * X["feature_2"] +
     1.5 * X["feature_3"] - 0.5 * X["feature_4"] +
     np.random.normal(0, 10, n_samples))

y_classify = np.random.randint(0, 2, n_samples)

# Add the target variable to the training dataset
train_df = X.copy()
train_df["target"] = y_classify

# Generate a separate dataset for predictions (without the target column)
# Using a different number of samples to avoid leaking information from
# training
n_pred_samples = 200
X_pred = pd.DataFrame({
    "feature_1": np.random.normal(50, 10, n_pred_samples),
    "feature_2": np.random.uniform(0, 100, n_pred_samples),
    "feature_3": np.random.binomial(100, 0.5, n_pred_samples),
    "feature_4": np.random.poisson(5, n_pred_samples),
    "feature_5": np.random.exponential(1, n_pred_samples)
})

# Save to CSV files
train_df.to_csv("train_regression_dataset.csv", index=False)
X_pred.to_csv("predict_regression_dataset.csv", index=False)
