from sklearn.model_selection import GridSearchCV

class HyperparameterTuner:
    def __init__(self, data_factory, model_factory):
        self.data_factory = data_factory
        self.models_to_train_and_tune = model_factory.get_models_to_train_and_tune()
        self.tuned_models = {}

    def tune_hyperparameters(self):
        for model_name, model in self.models_to_train_and_tune["models"].items():
            hyperparameters = self.models_to_train_and_tune["hyperparameters"][model_name]
            if hyperparameters:
                grid_search = GridSearchCV(model, hyperparameters, cv=5)
                grid_search.fit(self.data_factory.X_train, self.data_factory.y_train)
                self.tuned_models[model_name] = grid_search.best_estimator_
            else:
                model.fit(self.data_factory.X_train, self.data_factory.y_train)
                self.tuned_models[model_name] = model
        return self.tuned_models

