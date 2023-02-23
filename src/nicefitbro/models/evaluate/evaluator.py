import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class ModelEvaluator:
    def __init__(self, data_factory, trained_models):
        self.trained_models = trained_models
        self.data_factory = data_factory
        self.trained_model_performance = {}

    def evaluate_trained_models(self):
        for model_name, model in self.trained_models.items():
            y_val_pred = model.predict(self.data_factory.X_val)
            self.trained_model_performance[model_name] = {
                "R2": r2_score(self.data_factory.y_val, y_val_pred),
                "RMSE": np.sqrt(
                    mean_squared_error(self.data_factory.y_val, y_val_pred)
                ),
            }
        return self.trained_model_performance
