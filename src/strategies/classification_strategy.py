from abc import ABC, abstractmethod

class ClassificationStrategy(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def print_results(self, y_test, predictions):
        pass