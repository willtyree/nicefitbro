import abc
import boto3
import pandas as pd
from io import StringIO

class DataImporter(abc.ABC):
    """
    Abstract class for importing data.
    
    This class defines the common interface for importing data, and should be subclassed by concrete implementations
    that import data from specific sources, such as a local file, an S3 bucket, or a database.
    
    Attributes:
        None
        
    Methods:
        @abstractmethod
        import_data(source):
            Abstract method for importing data.
            - source: string indicating the source of the data.
            Returns: pandas DataFrame containing the imported data.
    """
    @abc.abstractmethod
    def import_data(self, *args, **kwargs):
        raise NotImplementedError
        
class LocalFileImporter(DataImporter):
    """
    Concrete implementation of the DataImporter abstract class for importing data from a local file.
    
    This class implements the import_data method for importing data from a local file using the pandas read_csv function.
    
    Attributes:
        None
        
    Methods:
        import_data(file_path):
            Imports data from a local file.
            - file_path: string indicating the path to the local file.
            Returns: pandas DataFrame containing the imported data.
    """
    def import_data(self, file_path, drop_cols=["Unnamed: 0", "api"]):
        df = pd.read_csv(file_path)

        if drop_cols:
            for col in drop_cols:
                if col in df.columns:
                    df.drop(columns=[col], axis=1, inplace=True)
        return df
    
class S3Importer(DataImporter):
    """
    Concrete implementation of the DataImporter abstract class for importing data from an S3 bucket.
    
    This class implements the import_data method for importing data from an S3 bucket using the boto3 library.
    
    Attributes:
        None
        
    Methods:
        import_data(client, bucket_name, file_name):
            Imports data from an S3 bucket.
            - client: an instantiated s3 client
            - bucket_name: string indicating the name of the S3 bucket.
            - file_name: string indicating the name of the file in the S3 bucket.
            Returns: pandas DataFrame containing the imported data.
    """
    def import_data(self, client, bucket_name, file_name, drop_cols=["Unnamed: 0", "api"]):
        csv_obj = client.get_object(
            Bucket=bucket_name,
            Key=file_name
        )
        body = csv_obj['Body']
        csv_string = body.read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_string))

        if drop_cols:
            for col in drop_cols:
                if col in df.columns:
                    df.drop(columns=[col], axis=1, inplace=True)
        return df