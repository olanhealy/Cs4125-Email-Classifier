from src.strategies.classification_strategy import ClassificationStrategy
class ModelContext:
    def __init__(self, strategy: ClassificationStrategy):
        self.strategy = strategy

    def train(self, X_train, y_train):
        self.strategy.train(X_train, y_train)

    def predict(self, X_test):
        return self.strategy.predict(X_test)

    def print_results(self, y_test, predictions):
        self.strategy.print_results(y_test, predictions)