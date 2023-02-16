import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from nicefitbro.data_prep.feature_engineering.feature_engineering_abc import FeatureEngineering

class FeatureTransformer(FeatureEngineering):
    """
    Concrete implementation of the FeatureEngineering abstract class for transforming features using various techniques.
    
    This class implements the engineer_features method for transforming features using the following techniques:
        - Polynomial Transformation
        - Log Transformation
        - Box Cox Transformation
        
    Attributes:
        degree (int): Degree of the polynomial transformation.
        method (str): Method to use for feature transformation.
            'polynomial': Use polynomial transformation.
            'log': Use log transformation.
            'box_cox': Use Box Cox transformation.
        feature (str or list of str): Feature(s) to apply the transformation to.
        
    Methods:
        engineer_features(data):
            Transforms the specified feature(s) of the data using the specified feature transformation technique.
            - data: pandas DataFrame containing the data.
            Returns: pandas DataFrame with the transformed features.
    """
    def __init__(self, features, degree=2, method='polynomial'):
        self.features = features
        self.degree = degree
        self.method = method
        
    def transform_features_polynomial(self, data, features):
        if len(features) == 1:
            feature_data = data[features].values.reshape(-1, 1)
        else:
            feature_data = data[features]
            
        poly = PolynomialFeatures(degree=self.degree)
        features_transformed = poly.fit_transform(feature_data)
        ft_transform_df = pd.DataFrame(features_transformed, columns=poly.get_feature_names_out(features))
        
        for col in ft_transform_df.columns:
            if col in data.columns:
                ft_transform_df.drop(columns=[col], axis=1, inplace=True)
                
        if "1" in ft_transform_df.columns:
            ft_transform_df.drop(columns=["1"], axis=1, inplace=True)
            
        data = pd.concat([data, ft_transform_df], axis=1)
            
        return data
    
    def transform_features_log(self, data, features):
        features_transformed = np.log1p(data[features])
        for col in features_transformed.columns:
            data[col] = features_transformed[col]
        return data
    
    def transform_features_box_cox(self, data, features):
        for col in features:
            data[col] = stats.boxcox(data[col])[0]
        return data
    
    def engineer_features(self, data, target=None):
        if self.features is None:
            raise ValueError("Feature must be specified for FeatureTransformer.")
        if self.method == 'polynomial':
            return self.transform_features_polynomial(data, self.features)
        elif self.method == 'log':
            return self.transform_features_log(data, self.features)
        elif self.method == 'box_cox':
            return self.transform_features_box_cox(data, self.features)