from sklearn.model_selection import train_test_split


class DataFactory:
    def __init__(self, data, target, features=None):
        self.features = features
        self.target = target
        if self.features:
            self.X = data[self.features]
            self.y = data[self.target]
        else:
            self.y = data[self.target]
            self.X = data.drop(columns=[self.target], axis=1)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.33, random_state=42
        )
