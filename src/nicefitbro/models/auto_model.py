from nicefitbro.models.factory.model_factory import ModelFactory
from nicefitbro.models.factory.data_factory import DataFactory
from nicefitbro.models.tune.tuner import HyperparameterTuner
from nicefitbro.models.train.trainer import ModelTrainer
from nicefitbro.models.evaluate.evaluator import ModelEvaluator


class AutoModel:
    def __init__(self, data, model_types, target, features=None):
        self.data_factory = DataFactory(data, target)
        self.model_factory = ModelFactory(model_types=model_types)
        self.tuner = HyperparameterTuner(self.data_factory, self.model_factory)

    def auto_model(self):
        tuned_models = self.tuner.tune_hyperparameters()
        mt = ModelTrainer(self.data_factory, tuned_models)
        trained_models = mt.train_models()
        me = ModelEvaluator(self.data_factory, trained_models)
        trained_model_performance = me.evaluate_trained_models()
        return trained_models, trained_model_performance
