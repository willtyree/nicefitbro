import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from nicefitbro.data_prep.feature_engineering.feature_engineering_abc import (
    FeatureEngineering,
)


class FeatureScaler(FeatureEngineering):
    """
    Concrete implementation of the FeatureEngineering abstract class for scaling features.

    This class implements the engineer_features method for scaling the features in the data using the StandardScaler or MinMaxScaler classes from scikit-learn.

    Attributes:
        method (str): String indicating the method to use for scaling the features.
            'standard': Scale features to have zero mean and unit variance using StandardScaler.
            'minmax': Scale features to have a minimum value of 0 and a maximum value of 1 using MinMaxScaler.

    Methods:
        engineer_features(data):
            Scales the features in the data.
            - data: pandas DataFrame containing the data.
            Returns: pandas DataFrame with scaled features.
    """

    def __init__(self, method="standard"):
        self.method = method

    def engineer_features(self, data, target=None):
        if self.method == "standard":
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
        elif self.method == "minmax":
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
        return pd.DataFrame(data_scaled, columns=data.columns)
