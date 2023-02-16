import numpy as np
import pandas as pd
from scipy import stats
from nicefitbro.data_prep.preprocess.preprocessor_abc import DataPreprocessor

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