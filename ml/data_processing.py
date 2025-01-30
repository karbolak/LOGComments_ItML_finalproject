import numpy as np
import pandas as pd
from ml.one_hot_encoder import OneHotEncoder


def detect_feature_types(df: pd.DataFrame, threshold: int = 2) -> dict:
    """
    Detects numerical and categorical features.

    Args:
        df (pd.DataFrame): The dataset.
        threshold (int): Max unique values for a numeric column to be
        categorical.

    Returns:
        dict: {"categorical": list of categorical features, "numerical": list
        of numerical features}
    """

    categorical_features = []
    numerical_features = []

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].nunique() <= threshold:
                categorical_features.append(column)
            else:
                numerical_features.append(column)
        else:
            categorical_features.append(column)

    return {"categorical": categorical_features,
            "numerical": numerical_features}


def preprocess_data(df: pd.DataFrame, feature_types: dict,
                    fit_encoders: bool = True, encoders: dict = None,
                    stats: dict = None):
    """
    Preprocess dataset: applies one-hot encoding to categorical features and
    Z-score normalization to numerical features.

    Args:
        df (pd.DataFrame): The dataset.
        feature_types (dict): Categorized feature names {"categorical": [...],
                                                         "numerical": [...]}
        fit_encoders (bool): Whether to fit new OneHotEncoders.
        encoders (dict): Pre-fitted encoders for transformation.
        stats (dict): Dictionary containing precomputed mean and std for
        numerical features.

    Returns:
        np.ndarray: Processed feature matrix.
        dict: Fitted encoders if `fit_encoders=True`, otherwise returns
        unchanged.
        dict: Fitted mean and std values if `fit_encoders=True`, otherwise
        returns unchanged.
    """
    processed_frames = []
    encoders = encoders or {}
    stats = stats or {}

    # One-hot encode categorical features
    for feature in feature_types["categorical"]:
        column_data = df[[feature]].values
        if fit_encoders:
            encoder = OneHotEncoder()
            encoder.fit(column_data)
            encoders[feature] = encoder
        else:
            encoder = encoders.get(feature)
            if encoder is None:
                raise ValueError(f"No encoder found for feature '{feature}'.")

        # Ensure test data doesn't introduce unseen categories
        for i, value in enumerate(column_data):
            # Check against trained categories
            if value not in encoder.categories_[0]:
                column_data[i] = encoder.unseen_category_

        transformed_data = encoder.transform(column_data)
        expanded_columns = [f"{feature}_{i}" for i in range(
            transformed_data.shape[1])]
        processed_frames.append(pd.DataFrame(transformed_data,
                                columns=expanded_columns))

    # Standardize numerical features (Z-score normalization)
    for feature in feature_types["numerical"]:
        column_data = df[[feature]].astype(float).values
        if fit_encoders:
            mean = np.mean(column_data, axis=0)
            std = np.std(column_data, axis=0)
            stats[feature] = (mean, std)
        else:
            if feature not in stats:
                raise ValueError(f"No scaling parameters found for feature"
                                 f" '{feature}'.")

            mean, std = stats[feature]

        if std == 0:
            std = 1  # Prevent division by zero
        standardized_data = (column_data - mean) / std
        processed_frames.append(pd.DataFrame(standardized_data,
                                columns=[feature]))

    return np.hstack([frame.values for frame in processed_frames]
                     ), encoders, stats


def stratified_train_validation_split(X, y, val_size=0.2, random_seed=42):
    """Perform a stratified split into training and validation sets."""
    np.random.seed(random_seed)
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    pos_split = int(len(pos_indices) * (1 - val_size))
    neg_split = int(len(neg_indices) * (1 - val_size))

    train_indices = np.concatenate([pos_indices[:pos_split],
                                    neg_indices[:neg_split]])
    val_indices = np.concatenate([pos_indices[pos_split:],
                                  neg_indices[neg_split:]])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]


def preprocess_datasets(train_dataset, test_dataset, target_column):
    """Preprocess training and testing datasets."""
    X_train_raw = train_dataset.drop(columns=[target_column])
    X_test_raw = test_dataset.drop(columns=[target_column])

    # Detect feature types
    feature_types = detect_feature_types(X_train_raw)

    # Extract target values
    y_train = train_dataset[target_column].values
    y_test = test_dataset[target_column].values

    # Preprocess TRAINING dataset (fit encoders & compute stats)
    X_train, encoders, stats = preprocess_data(X_train_raw, feature_types,
                                               fit_encoders=True)

    # Preprocess TEST dataset (reuse encoders & scaling stats)
    X_test, _, _ = preprocess_data(X_test_raw, feature_types,
                                   fit_encoders=False, encoders=encoders,
                                   stats=stats)

    return X_train, X_test, y_train, y_test
