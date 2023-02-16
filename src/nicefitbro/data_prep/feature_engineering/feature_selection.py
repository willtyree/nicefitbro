import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from nicefitbro.data_prep.feature_engineering.feature_engineering_abc import (
    FeatureEngineering,
)


class FeatureSelection(FeatureEngineering):
    """
    Concrete implementation of the FeatureEngineering abstract class for selecting features using various algorithms.

    This class implements the select_features method for selecting features using the following algorithms:
        - SelectKBest
        - Recursive Feature Elimination (RFE)
        - Lasso Regression

    Attributes:
        model (sklearn estimator): Sklearn estimator to use for feature selection.
        k (int): Number of features to keep in SelectKBest algorithm.
        threshold (float): Threshold for determining which features to keep based on the Lasso Regression algorithm.
        method (str): Method to use for feature selection.
            'select_k_best': Use SelectKBest algorithm.
            'rfe': Use Recursive Feature Elimination (RFE) algorithm.
            'lasso': Use Lasso Regression algorithm.
            'manual': Manually select the columns given a list

    Methods:
        engineer_features(data, target):
            Selects the most relevant features from the data using the specified feature selection algorithm.
            - data: pandas DataFrame containing the data.
            - target: pandas Series containing the target variable.
            Returns: pandas DataFrame with the selected features.
    """

    def __init__(self, k=15, threshold=0.5, method="select_k_best"):
        self.k = k
        self.threshold = threshold
        self.method = method

    def manual_selection(self, data, features, target):
        """Preforms manual feature selection on the input dataframe
        Args:
            input_df (pd.DataFrame): input tabular data
            features (list[str]): subset of columns to be selected (X)
            target (str): target colum to be predicted (y)
        Raises:
            Exception: selected features do no exist
            Exception: selected target does not exist
        Returns:
            pd.DataFrame: dataframe with selected features and target
        """
        # data checks
        for f in features:
            if f not in data.columns:
                raise Exception

        if target not in data.columns:
            raise Exception

        return data[[target] + features]

    def select_features_select_k_best(self, data, target):
        feature_df = data.drop(columns=[target])
        target_df = data[target]
        selector = SelectKBest(f_regression, k=self.k)
        selector.fit(feature_df, target_df)
        mask = selector.get_support()
        selected_features = feature_df.columns[mask]
        return data[selected_features]

    def select_features_rfe(self, data, target):
        """
        warning: takes a while to compute
        """
        feature_df = data.drop(columns=[target])
        target_df = data[target]
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, self.k)
        selector.fit(feature_df, target_df)
        mask = selector.support_
        selected_features = feature_df.columns[mask]
        return data[selected_features]

    def select_features_lasso(self, data, target):
        feature_df = data.drop(columns=[target])
        target_df = data[target]
        selector = LassoCV(cv=5, random_state=0)
        selector.fit(feature_df, target_df)
        mask = selector.coef_ != 0
        selected_features = feature_df.columns[mask]
        return data[selected_features]

    def engineer_features(self, data, target, features=None):
        if self.method == "select_k_best":
            return self.select_features_select_k_best(data, target)
        elif self.method == "rfe":
            return self.select_features_rfe(data, target)
        elif self.method == "lasso":
            return self.select_features_lasso(data, target)
        elif self.method == "manual":
            return self.manual_selection(data, features, target)
        else:
            raise ValueError(
                "Invalid method for feature selection. Choose 'select_k_best', 'rfe', 'lasso', or 'manual'."
            )
