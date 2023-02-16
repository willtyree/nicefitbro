import abc
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, RFE, f_regression

class FeatureEngineering(abc.ABC):
    """
    Abstract class for feature engineering.
    
    This class provides a blueprint for implementing feature engineering for machine learning models.
    
    Methods:
        engineer_features(data):
            Creates new features by transforming the existing features.
            - data: pandas DataFrame containing the data.
            Returns: pandas DataFrame with the engineered features.
    """    
    @abc.abstractmethod
    def engineer_features(self, data):
        raise NotImplementedError

class CorrelationAnalysis(FeatureEngineering):
    """
    Concrete implementation of the FeatureEngineering abstract class for selecting features using correlation analysis.
    
    This class implements the select_features method for selecting features by calculating the correlation between features and the target variable, and retaining only the features with a high correlation.
    
    Attributes:
        target_col (str): Name of the target column.
        threshold (float): Threshold for determining which features to keep based on the correlation with the target variable.
        method (str): Method to use for correlation analysis.
            'pearson': Use Pearson correlation.
            'spearman': Use Spearman correlation.
            'kendall': Use Kendall correlation.
        
    Methods:
        engineer_features(data):
            Selects the most relevant features from the data using correlation analysis.
            - data: pandas DataFrame containing the data.
            Returns: pandas DataFrame with the selected features.
    """
    def __init__(self, target_col, threshold=0.1, method='pearson'):
        self.target_col = target_col
        self.threshold = threshold
        self.method = method
        
    def calculate_correlation_pearson(self, data):
        corr = data.corr()[self.target_col].abs()
        features = corr[corr > self.threshold].index
        return data[features]
    
    def calculate_correlation_spearman(self, data):
        corr = data.corr(method="spearman")[self.target_col].abs()
        features = corr[corr > self.threshold].index
        return data[features]
    
    def calculate_correlation_kendall(self, data):
        corr = data.corr(method="kendall")[self.target_col].abs()
        features = corr[corr > self.threshold].index
        return data[features]
    
    def engineer_features(self, data, target=None):
        if self.method == 'pearson':
            return self.calculate_correlation_pearson(data)
        elif self.method == 'spearman':
            return self.calculate_correlation_spearman(data)
        elif self.method == 'kendall':
            return self.calculate_correlation_kendall(data)

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
        
    Methods:
        engineer_features(data, target):
            Selects the most relevant features from the data using the specified feature selection algorithm.
            - data: pandas DataFrame containing the data.
            - target: pandas Series containing the target variable.
            Returns: pandas DataFrame with the selected features.
    """
    def __init__(self, k=15, threshold=0.5, method='select_k_best'):
        self.k = k
        self.threshold = threshold
        self.method = method
        
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
    
    def engineer_features(self, data, target):
        if self.method == 'select_k_best':
            return self.select_features_select_k_best(data, target)
        elif self.method == 'rfe':
            return self.select_features_rfe(data, target)
        elif self.method == 'lasso':
            return self.select_features_lasso(data, target)


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
    def __init__(self, method='standard'):
        self.method = method
        
    def engineer_features(self, data, target=None):
        if self.method == 'standard':
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
        elif self.method == 'minmax':
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
        return pd.DataFrame(data_scaled, columns=data.columns)
    
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
    def __init__(self, method='ordinal', columns=None):
        self.method = method
        self.columns = columns
        
    def engineer_features(self, data, target=None):
        if self.method == 'ordinal':
            encoder = OrdinalEncoder()
            data[self.columns] = encoder.fit_transform(data[self.columns])
        elif self.method == 'onehot':
            encoder = OneHotEncoder()
            onehot_encoded = encoder.fit_transform(data[self.columns])
            onehot_encoded_df = pd.DataFrame(onehot_encoded.toarray(), columns=encoder.get_feature_names(self.columns))
            data = pd.concat([data.drop(columns=self.columns), onehot_encoded_df], axis=1)
        return data