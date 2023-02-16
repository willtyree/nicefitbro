import abc

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