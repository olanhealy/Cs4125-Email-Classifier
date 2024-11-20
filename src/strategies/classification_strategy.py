from abc import ABC, abstractmethod

class ClassificationStrategy(ABC):
    """
    Abstract base class for classification strategies.
    Defines a common interface for different machine learning models to ensure consistency.
    """
    def __init__(self, model):
        """
        Initialise the strategy with a machine learning model.

        :param model: An instance of a machine learning model created by factory pattern.
        """
        self.model = model

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Abstract method to train the model with the given training data.

        :param X_train: Features for training.
        :param y_train: Labels for training.
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Abstract method to predict labels for the given test data.

        :param X_test: Features for prediction.
        :return: Predicted labels.
        """
        pass

    @abstractmethod
    def print_results(self, y_test, predictions):
        """
        Abstract method to print classification results.

        :param y_test: True labels.
        :param predictions: Predicted labels.
        """
        pass