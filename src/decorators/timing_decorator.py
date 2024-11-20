#src/decorators/timing_decorator.py
import time
from src.decorators.decorator import ClassifierDecorator

class TimingDecorator(ClassifierDecorator):
    """
        A decorator class that adds timing functionality.
        This decorator logs the time taken for training, prediction and printing results..
    """

    def __init__(self, strategy):
        """
        Initialise the timing decorator.
        """
        super().__init__(strategy)

    def train(self, X_train, y_train) -> None:
        """
            Measure and log the time taken to train the model.

            :return: The training result from the wrapped strategy.
        """
        start_time = time.time()
        result = super().train(X_train, y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[TIME] Training took {elapsed_time:.4f} seconds")
        return result

    def predict(self, X_test) -> int:
        """
            Measure and log the time taken to make a prediction.

            :return: The prediction result from the wrapped strategy.
        """
        start_time = time.time()
        predictions = super().predict(X_test)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[TIME]Prediction took {elapsed_time:.4f} seconds")
        return predictions

    def print_results(self, y_test, predictions):
        """
            Measure and print the time taken to print results.
        """
        start_time = time.time()
        super().print_results(y_test, predictions)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[TIME] Printing results took {elapsed_time:.4f} seconds.")
