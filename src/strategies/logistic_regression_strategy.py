from sklearn.metrics import classification_report
from src.strategies.classification_strategy import ClassificationStrategy

class LogisticRegressionStrategy(ClassificationStrategy):
    def __init__(self, model):
        super().__init__(model)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def print_results(self, y_test, predictions):
        print(classification_report(y_test, predictions))