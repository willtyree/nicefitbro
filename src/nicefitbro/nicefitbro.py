from nicefitbro.ingestors.local_ingestor import LocalFileIngestor
from nicefitbro.preprocess.missing_value_processor import MissingValuePreprocessor
from nicefitbro.preprocess.outlier_detector import OutlierDetector
from nicefitbro.feature_engineering.categorical_encoding import CategoricalEncoder
from nicefitbro.feature_engineering.correlation_analysis import CorrelationAnalysis
from nicefitbro.feature_engineering.feature_scaling import FeatureScaler
from nicefitbro.feature_engineering.feature_selection import FeatureSelection
from nicefitbro.feature_engineering.feature_transformations import FeatureTransformer
from nicefitbro.pipeliners.fe_pipeliner import FtEngineeringPipeliner
from nicefitbro.pipeliners.preprocessor_pipeliner import PreprocessorPipeliner
from nicefitbro.pipeliners.prepper import DataPrepper
from nicefitbro.models.auto_model import AutoModel
from nicefitbro.config.run_config import RunConfig


class NiceFitBro:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self.preprocessor = None
        self.engineer = None
        self.processor_steps = []
        self.feature_engineering_steps = []
        self.local_file_ingestor = LocalFileIngestor()

    def _missing(self):
        if self.run_config.missing_value_method:
            missing_value_preprocessor = MissingValuePreprocessor(
                method=self.run_config.missing_value_method
            )
            self.processor_steps.append(missing_value_preprocessor)

    def _outliers(self):
        if self.run_config.outlier_detector_method:
            outlier_detector = OutlierDetector(
                method=self.run_config.outlier_detector_method
            )
            self.processor_steps.append(outlier_detector)

    def _selectors(self):
        if self.run_config.feature_selector_k:
            feature_selector = FeatureSelection(k=self.run_config.feature_selector_k)
            self.feature_engineering_steps.append(feature_selector)

    def _transformers(self):
        if self.run_config.feature_transformer_method:
            feature_transformer = FeatureTransformer(
                features=self.run_config.feature_transformer_features,
                method=self.run_config.feature_transformer_method,
            )
            self.feature_engineering_steps.append(feature_transformer)

    def _scalers(self):
        if self.run_config.feature_scaler_method:
            feature_scaler = FeatureScaler(method=self.run_config.feature_scaler_method)
            self.feature_engineering_steps.append(feature_scaler)

    def _preprocess(self):
        self._missing()
        self._outliers()

        if self.processor_steps:
            self.preprocessor = PreprocessorPipeliner(self.processor_steps)

    def _engineer(self):
        self._selectors()
        self._transformers()
        self._scalers()

        if self.feature_engineering_steps:
            self.engineer = FtEngineeringPipeliner(self.feature_engineering_steps)

    def prepare_data(self):
        self._preprocess()
        self._engineer()

        data_prepper = DataPrepper(
            self.local_file_ingestor,
            self.run_config.target,
            self.preprocessor,
            self.engineer,
        )
        processed_data = data_prepper.load_and_preprocess_data(
            self.run_config.file_path
        )
        return processed_data

    def autofit(self, processed_data):
        am = AutoModel(
            processed_data, self.run_config.model_types, self.run_config.target
        )
        trained_models, performance = am.auto_model()
        return trained_models, performance

    def sendit(self):
        return self.autofit(self.prepare_data())
