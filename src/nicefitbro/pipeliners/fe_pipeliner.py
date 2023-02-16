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
