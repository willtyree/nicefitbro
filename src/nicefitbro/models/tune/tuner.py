from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

class HyperparameterTuner:
    def __init__(self, data, model_factory, target, features=None):
        self.data = data
        self.model_factory = model_factory
        self.target = target
        self.features = features
        if self.features:
            self.X = data[self.features]
            self.y = data[self.target]
        else:
            self.y = data[self.target]
            self.X = data.drop(columns=[self.target], axis=1)
        self.trained_models = {}
        self.models_to_train_and_tune = model_factory.get_models_to_train_and_tune()
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.33, random_state=42
        )

    def tune_hyperparameters(self):
        for model_name, model in self.models_to_train_and_tune["models"].items():
            print(model_name)
            hyperparameters = self.models_to_train_and_tune["hyperparameters"][model_name]
            if hyperparameters:
                grid_search = GridSearchCV(model, hyperparameters, cv=5)
                grid_search.fit(self.X_train, self.y_train)
                self.trained_models[model_name] = grid_search.best_estimator_
            else:
                model.fit(self.X_train, self.y_train)
                self.trained_models[model_name] = model

    def train_models(self):
        for model_name, model in self.trained_models.items():
            model.fit(self.X_train, self.y_train)

    def get_trained_models(self):
        return self.trained_models
    
    def train_and_tune_models(self):
        self.tune_hyperparameters()
        self.train_models()
        return self.get_trained_models()