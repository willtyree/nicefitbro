class ModelTrainer:
    def __init__(self, data_factory, tuned_models):
        self.data_factory = data_factory
        self.tuned_models = tuned_models
        self.trained_models = {}

    def train_models(self):
        for model_name, model in self.tuned_models.items():
            model.fit(self.data_factory.X_train, self.data_factory.y_train)
        self.trained_models = self.tuned_models
        return self.trained_models
