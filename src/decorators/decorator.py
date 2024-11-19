#src/decorators/decorator
from abc import ABC
from src.strategies.classification_strategy import ClassificationStrategy

class ClassifierDecorator(ClassificationStrategy, ABC):
    """
        A base decorator class that wraps around a ClassificationStrategy object.
        This allows for dynamically adding functionalities to existing strategies.
    """

    def __init__(self, strategy: ClassificationStrategy):
        """
            Initialise the decorator with a strategy instance.

            :param strategy: An instance of a class implementing the ClassificationStrategy interface.
        """
        self._strategy = strategy

    def train(self, X_train, y_train) -> None:
        """
            This method calls the train() method of the wrapped strategy.
        """
        self._strategy.train(X_train, y_train)

    def predict(self, X_test) -> int:
        """
            This method calls the predict() method of the wrapped strategy and returns its result.

            :return: The prediction result from the wrapped strategy.
        """
        return self._strategy.predict(X_test)

    def print_results(self, y_test, predictions):
        """
            This method calls the print_results() method of the wrapped strategy and returns its result.

            :return: The printed result from the wrapped strategy.
        """
        return self._strategy.print_results(y_test, predictions)
