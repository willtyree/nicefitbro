import pandas as pd
from io import StringIO
from nicefitbro.ingestors.ingestor_abc import DataIngestor


class S3Ingestor(DataIngestor):
    """
    Concrete implementation of the DataIngestor abstract class for importing data from an S3 bucket.

    This class implements the ingest_data method for importing data from an S3 bucket using the boto3 library.

    Attributes:
        None

    Methods:
        ingest_data(client, bucket_name, file_name):
            Imports data from an S3 bucket.
            - client: an instantiated s3 client
            - bucket_name: string indicating the name of the S3 bucket.
            - file_name: string indicating the name of the file in the S3 bucket.
            Returns: pandas DataFrame containing the imported data.
    """

    def ingest_data(
        self, client, bucket_name, file_name, drop_cols=["Unnamed: 0", "api"]
    ):
        csv_obj = client.get_object(Bucket=bucket_name, Key=file_name)
        body = csv_obj["Body"]
        csv_string = body.read().decode("utf-8")
        df = pd.read_csv(StringIO(csv_string))

        if drop_cols:
            for col in drop_cols:
                if col in df.columns:
                    df.drop(columns=[col], axis=1, inplace=True)
        return df
