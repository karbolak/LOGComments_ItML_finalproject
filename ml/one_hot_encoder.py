import numpy as np


class OneHotEncoder:
    """Custom implementation of OneHotEncoder.

    Attributes:
        categories_ (dict): Dictionary containing unique categories for each
        column.
        unseen_category_ (str): Placeholder for unseen categories.
    """

    def __init__(self):
        self.categories_ = {}
        self.unseen_category_ = "<UNSEEN>"

    def fit(self, X: np.ndarray) -> None:
        """Fits the encoder to the unique categories in X.

        Args:
            X (np.ndarray): 2D array of input categorical data.
        """
        if X.ndim != 2:
            raise ValueError("Input must be a 2D array.")

        for col in range(X.shape[1]):
            # Convert all values to string for consistent comparison
            column_data = X[:, col].astype(str)
            unique_values = np.unique(column_data)
            self.categories_[col] = np.append(unique_values,
                                              self.unseen_category_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms the input data into one-hot encoded format.

        Args:
            X (np.ndarray): 2D array of input categorical data.

        Returns:
            np.ndarray: One-hot encoded array.
        """
        if not self.categories_:
            raise ValueError("Encoder has not been fitted.")

        if X.ndim != 2:
            raise ValueError("Input must be a 2D array.")

        encoded_arrays = []
        for col in range(X.shape[1]):
            column_data = X[:, col].astype(str)
            categories = self.categories_.get(col, None)

            if categories is None:
                raise ValueError(f"Column {col} was not fitted.")

            # Create a binary matrix for the column
            column_encoded = np.zeros((len(column_data), len(categories)))
            for i, value in enumerate(column_data):
                if value not in categories:
                    value = self.unseen_category_
                column_encoded[i, np.where(categories == value)[0][0]] = 1

            encoded_arrays.append(column_encoded)

        # Concatenate all encoded columns
        return np.hstack(encoded_arrays)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits the encoder and transforms the input data.

        Args:
            X (np.ndarray): 2D array of input categorical data.

        Returns:
            np.ndarray: One-hot encoded array.
        """
        self.fit(X)
        return self.transform(X)
