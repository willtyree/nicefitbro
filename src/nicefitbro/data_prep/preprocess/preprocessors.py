import abc
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor

class DataPreprocessor(abc.ABC):
    """
    Abstract class for preprocessing data.
    
    This class defines the common interface for preprocessing data, and should be subclassed by concrete implementations
    that perform specific preprocessing tasks, such as handling missing values, scaling features, and encoding
    categorical variables.
    
    Attributes:
        None
        
    Methods:
        @abstractmethod
        preprocess_data(data):
            Abstract method for preprocessing data.
            - data: pandas DataFrame containing the data to preprocess.
            Returns: pandas DataFrame containing the preprocessed data.
    """
    @abc.abstractmethod
    def preprocess_data(self, data, *args, **kwargs):
        raise NotImplementedError


class MissingValuePreprocessor(DataPreprocessor):
    """
    Concrete implementation of the DataPreprocessor abstract class for handling missing values.
    
    This class implements the preprocess_data method for handling missing values in the data.
    
    Attributes:
        method (str): String indicating the method to use for handling missing values.
            'mean': Replace missing values with the mean of the column.
            'median': Replace missing values with the median of the column.
            'fill': Replace missing values with input fill_value. Defualts to 0
            'drop': Drop rows with missing values.
        
    Methods:
        preprocess_data(data):
            Handles missing values in the data.
            - data: pandas DataFrame containing the data.
            Returns: pandas DataFrame with missing values handled.
    """
    def __init__(self, method='mean'):
        self.method = method
        
    def preprocess_data(self, data, fill_value=0):
        if self.method == 'mean':
            data.fillna(data.mean(), inplace=True)
        elif self.method == "median":
            data.fillna(data.median(), inplace=True)
        elif self.method == 'fill':
            data.fillna(fill_value, inplace=True)
        elif self.method == 'drop':
            data.dropna(inplace=True)
        return data

class OutlierDetector(DataPreprocessor):
    """
    Concrete implementation of the DataPreprocessor abstract class for detecting outliers.
    
    This class implements the preprocess_data method for detecting outliers in the data using multiple methods, including the ZScore method, IQR method, Mahalanobis Distance method, and Local Outlier Factor (LOF) method.
    
    Attributes:
        method (str): String indicating the method to use for detecting outliers.
            'zscore': Detect outliers using the ZScore method.
            'iqr': Detect outliers using the IQR method.
            'mahalanobis': Detect outliers using the Mahalanobis Distance method.
            'lof': Detect outliers using the Local Outlier Factor (LOF) method.
        threshold (float): Threshold for determining outliers. The specific meaning of this threshold will depend on the method used for outlier detection.
        
    Methods:
        preprocess_data(data):
            Detects outliers in the data.
            - data: pandas DataFrame containing the data.
            Returns: pandas DataFrame without outlier data points.
    """
    def __init__(self, method='zscore'):
        self.method = method
        
    def detect_outliers_zscore(self, data):
        return data[(np.abs(stats.zscore(data)) < 3).any(axis=1)]
    
    def detect_outliers_iqr(self, data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        return data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    def detect_outliers_mahalanobis(self, data):
        mean = data.mean()
        covariance = data.cov()
        inv_covariance = np.linalg.inv(covariance)
        data_minus_mean = data - mean
        mahalanobis_distance = np.sqrt(np.sum(data_minus_mean.dot(inv_covariance) * data_minus_mean, axis=1))
        return data[mahalanobis_distance < 3]
    
    def detect_outliers_lof(self, data):
        lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
        outliers = lof.fit_predict(data)
        return data[outliers == 1]
    
    def preprocess_data(self, data):
        if self.method == 'zscore':
            return self.detect_outliers_zscore(data)
        elif self.method == 'iqr':
            return self.detect_outliers_iqr(data)
        elif self.method == 'mahalanobis':
            return self.detect_outliers_mahalanobis(data)
        elif self.method == 'lof':
            return self.detect_outliers_lof(data)
        else:
            raise ValueError("Invalid method for outlier detection. Choose 'zscore', 'iqr', 'mahalanobis', or 'lof'.")