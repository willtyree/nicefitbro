import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

class ModelFactory:
    def __init__(self, model_types):
        self.model_types = model_types
        self.model_options = {
            "lr": LinearRegression(),
            "ridge": Ridge(),
            "lasso": Lasso(),
            "elastic": ElasticNet(),
            "bayesridge": BayesianRidge(),
            "sgd": SGDRegressor(),
            "knn": KNeighborsRegressor(),
            "gpr": GaussianProcessRegressor(),
            "dtr": DecisionTreeRegressor(),
            "rfr": RandomForestRegressor(),
            "gbr": GradientBoostingRegressor(),
            "xgb": xgb.XGBRegressor(),
            "poly": Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ])
        }
        self.hyperparameter_options = {
            "lr": {},
            "ridge": {"alpha": [0.1, 1.0, 10.0]},
            "lasso": {"alpha": [0.1, 1.0, 10.0]},
            "elastic": {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 1.0]},
            "bayesridge": {},
            "sgd": {"loss": ["squared_loss", "huber"], "alpha": [0.1, 0.01, 0.001]},
            "knn": {"n_neighbors": [3, 5, 7]},
            "gpr": {},
            "dtr": {"max_depth": [3, 5, 7]},
            "rfr": {"n_estimators": [50, 100, 150], "max_depth": [3, 5, 7]},
            "gbr": {"n_estimators": [50, 100, 150], "max_depth": [3, 5, 7]},
            "xgb": {"n_estimators": [50, 100, 150], "max_depth": [3, 5, 7]},
            "poly": {}
        }
        self.models = {}
        self.hyperparameters = {}
        for model_type in model_types:
            if model_type in self.model_options.keys():
                self.models[model_type] = self.model_options[model_type]
                self.hyperparameters[model_type] = self.hyperparameter_options[model_type]

    def get_models_to_train_and_tune(self):
        return {"models": self.models, "hyperparameters": self.hyperparameters}