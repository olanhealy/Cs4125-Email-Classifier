#src/decorators/error_handling_decorator.py
from src.decorators.decorator import ClassifierDecorator

class ErrorHandlingDecorator(ClassifierDecorator):
    """
    A decorator class that adds error handling functionality.
    This decorator logs any exceptions that occur during training, prediction and printing results.
    """

    def __init__(self, strategy):
        """
        Initialise the error catching decorator.
        """
        super().__init__(strategy)

    def train(self, X_train, y_train) -> None:
        """
        Wrap the train() method to catch and log any exceptions that occur.
        If an exception occurs, it logs the error but re-raises it.

        :return: The training result from the wrapped strategy, or -1 if an error occurs.
        """
        try:
            print("[DEBUG] Training with ErrorHandlingDecorator...")
            return super().train(X_train, y_train)
        except Exception as e:
            model_name = self._strategy.__class__.__name__
            print(f"[ERROR] An error occurred during training for {model_name}: {e}")
            raise

    def predict(self, X_test) -> int:
        """
        Wrap the predict() method to catch and log any exceptions that occur.
        If an exception occurs, it logs the error but re-raises it.

        :return: The prediction result from the wrapped strategy, or -1 if an error occurs.
        """
        try:
            print("[DEBUG] Predicting with ErrorHandlingDecorator...")
            return super().predict(X_test)
        except Exception as e:
            model_name = self._strategy.__class__.__name__
            print(f"[ERROR] An error occurred during prediction for {model_name}: {e}")
            raise

    def print_results(self, y_test, predictions):
        """
            Wrap the print_results() method to catch and log any exceptions that occur.
            If an exception occurs, it logs the error but re-raises it.

            :return: The printed results from the wrapped strategy, or -1 if an error occurs.
        """
        try:
            print("[DEBUG] Printing results with ErrorHandlingDecorator...")
            return super().print_results(y_test, predictions)
        except Exception as e:
            print(f"[ERROR] An error occurred while printing results: {e}")
            raise
