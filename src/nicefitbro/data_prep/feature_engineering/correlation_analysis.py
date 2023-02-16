from nicefitbro.data_prep.feature_engineering.feature_engineering_abc import FeatureEngineering


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