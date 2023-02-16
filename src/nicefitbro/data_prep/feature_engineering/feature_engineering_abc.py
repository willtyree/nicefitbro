
import abc

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