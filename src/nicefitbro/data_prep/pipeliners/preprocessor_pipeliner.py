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