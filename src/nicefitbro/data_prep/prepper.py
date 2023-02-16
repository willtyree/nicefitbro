import mlflow

class PreprocessorPipeliner:
    """
    Class for organizing and executing a pipeline of preprocessing steps.
    
    This class takes a list of preprocessing steps as input, and performs those steps in the order specified by the user.
    
    Attributes:
        preprocessor_steps (list): A list of preprocessing steps to perform. Each step should be an instance of a concrete
        implementation of the DataPreprocessor abstract class.
        
    Methods:
        preprocess_data(data):
            Executes the preprocessing pipeline on the input data.
            - data: pandas DataFrame containing the data to preprocess.
            Returns: pandas DataFrame containing the preprocessed data.
    """
    def __init__(self, preprocessor_steps):
        self.preprocessor_steps = preprocessor_steps
        
    def preprocess_data(self, data):
        for step in self.preprocessor_steps:
            data = step.preprocess_data(data)
        return data
    
class FtEngineeringPipeliner:
    """
    Class for organizing and executing a pipeline of feature engineering steps.
    
    This class takes a list of feature engineering steps as input, and performs those steps in the order specified by the user.
    
    Attributes:
        fe_steps (list): A list of preprocessing steps to perform. Each step should be an instance of a concrete
        implementation of the FeatureEngineering abstract class.
        
    Methods:
        engineer_features(data):
            Executes the feature engineering pipeline on the input data.
            - data: pandas DataFrame containing the data to engineer.
            Returns: pandas DataFrame containing the engineered data.
    """
    def __init__(self, fe_steps):
        self.fe_steps = fe_steps
        
    def engineer_features(self, data, target):
        for step in self.fe_steps:
            data = step.engineer_features(data, target)
        return data
    
class DataPrepper:
    """
    Class for loading and preprocessing data.
    
    This class combines a data importer and a data preprocessor into a single object, and provides a convenient way
    to load and preprocess data in one step.
    
    Attributes:
        importer (DataImporter): An instance of a concrete implementation of the DataImporter abstract class.
        preprocessor (DataPreprocessor): An instance of a concrete implementation of the DataPreprocessor abstract class.
        engineer (FtEngineeringPipeliner): 
        target (str): String value of the target column name
        
    Methods:
        load_and_preprocess_data(source):
            Loads and preprocesses data in one step.
            - source: string indicating the source of the data, depending on the importer implementation.
            Returns: pandas DataFrame containing the preprocessed data.
    """
    def __init__(self, importer, preprocessor, engineer, target):
        self.importer = importer
        self.preprocessor = preprocessor
        self.engineer = engineer
        self.target = target
        
    def load_and_preprocess_data(self, source):
        data = self.importer.import_data(source)
        preprocessed_data = self.preprocessor.preprocess_data(data)
        engineered_data = self.engineer.engineer_features(preprocessed_data, self.target)
        return engineered_data

# Wrap the DataPrepper class in a mlflow.pyfunc object
class DataPrepperPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(self, data_prepper):
        self.data_prepper = data_prepper

    def predict(self, context, model_input):
        return self.data_prepper.load_and_preprocess_data(model_input)