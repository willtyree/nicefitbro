import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from nicefitbro.feature_engineering.feature_engineering_abc import (
    FeatureEngineering,
)


class CategoricalEncoder(FeatureEngineering):
    """
    Concrete implementation of the FeatureEngineering abstract class for encoding categorical variables.

    This class implements the engineer_features method for encoding categorical variables in the data using the OrdinalEncoder or OneHotEncoder classes from scikit-learn.

    Attributes:
        method (str): String indicating the method to use for encoding the categorical variables.
            'ordinal': Encode categorical variables as integers using OrdinalEncoder.
            'onehot': Encode categorical variables as one-hot encoded binary variables using OneHotEncoder.
        columns (list): List of column names in the data to encode as categorical variables.

    Methods:
        engineer_features(data):
            Encodes the categorical variables in the data.
            - data: pandas DataFrame containing the data.
            Returns: pandas DataFrame with encoded categorical variables.
    """

    def __init__(self, method="ordinal", columns=None):
        self.method = method
        self.columns = columns

    def engineer_features(self, data, target=None):
        # ensure features are of type object
        cat_cols = [col for col in data.columns if data[col].dtype == "object"]

        if self.method == "ordinal":
            encoder = OrdinalEncoder()
            data[cat_cols] = encoder.fit_transform(data[cat_cols])
        elif self.method == "onehot":
            encoder = OneHotEncoder(sparse=False)
            onehot_encoded = encoder.fit_transform(data[cat_cols])
            onehot_encoded_df = pd.DataFrame(
                onehot_encoded.toarray(), columns=encoder.get_feature_names(cat_cols)
            )
            data = pd.concat([data.drop(columns=cat_cols), onehot_encoded_df], axis=1)
        return data
