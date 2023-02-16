import pandas as pd
from nicefitbro.ingestors.ingestor_abc import DataIngestor


class LocalFileIngestor(DataIngestor):
    """
    Concrete implementation of the DataIngestor abstract class for importing data from a local file.

    This class implements the ingest_data method for importing data from a local file using the pandas read_csv function.

    Attributes:
        None

    Methods:
        ingest_data(file_path):
            Imports data from a local file.
            - file_path: string indicating the path to the local file.
            Returns: pandas DataFrame containing the imported data.
    """

    def ingest_data(self, file_path, drop_cols=["Unnamed: 0", "api"]):
        df = pd.read_csv(file_path)

        if drop_cols:
            for col in drop_cols:
                if col in df.columns:
                    df.drop(columns=[col], axis=1, inplace=True)
        return df
