import mlflow


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
        data = self.importer.ingest_data(source)
        preprocessed_data = self.preprocessor.preprocess_data(data)
        engineered_data = self.engineer.engineer_features(
            preprocessed_data, self.target
        )
        return engineered_data


# Wrap the DataPrepper class in a mlflow.pyfunc object
class DataPrepperPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(self, data_prepper):
        self.data_prepper = data_prepper

    def predict(self, context, model_input):
        return self.data_prepper.load_and_preprocess_data(model_input)
