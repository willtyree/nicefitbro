import abc


class DataIngestor(abc.ABC):
    """
    Abstract class for ingesting data.

    This class defines the common interface for ingesting data, and should be subclassed by concrete implementations
    that ingest data from specific sources, such as a local file, an S3 bucket, or a database.

    Attributes:
        None

    Methods:
        @abstractmethod
        ingest_data(source):
            Abstract method for ingesting data.
            - source: string indicating the source of the data.
            Returns: pandas DataFrame containing the ingested data.
    """

    @abc.abstractmethod
    def ingest_data(self, source):
        raise NotImplementedError
