from typing import List, Union
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd
from pandas.api.types import CategoricalDtype


def detect_feature_types(data: Union[Dataset, pd.DataFrame],
                         threshold: int = 10) -> List[Feature]:
    """
    Detects feature types (categorical or numerical) in a dataset or DataFrame.
    Numeric features with limited unique values are classified as categorical.

    Args:
        data (Union[Dataset, pd.DataFrame]): The dataset/DataFrame to analyze.
        threshold (int): Max unique values for numeric columns to be
                         considered categorical.

    Returns:
        List[Feature]: A list of features with detected types.
    """
    # Check if input is Dataset, if so, read it as DataFrame
    if isinstance(data, Dataset):
        df = data.read()
    else:
        df = data

    # Initialize an empty list to store the feature objects
    features = []

    # Iterate through the columns of the DataFrame
    for column in df.columns:
        # Determine the type of feature
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].nunique() <= threshold:
                # Limited unique values in numeric column
                feature_type = "categorical"
            else:
                feature_type = "numerical"
        elif (isinstance(df[column].dtype, CategoricalDtype)
              or pd.api.types.is_object_dtype(df[column])):
            feature_type = "categorical"
        else:
            raise ValueError(f"Unknown data type for column {column}")

        # Create a Feature object with values and append it to the list
        features.append(Feature(name=column, type=feature_type,
                        values=df[column].values))

    return features
