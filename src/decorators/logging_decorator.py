#src/decorators/logging_decorator.py
from src.decorators.decorator import ClassifierDecorator

class LoggingDecorator(ClassifierDecorator):
    """
    A decorator class that adds logging functionality.
    This decorator logs during training, prediction and printing results.
    """

    def train(self, X_train, y_train) -> None:
        """
            Log the start and completion of the training process.

            :return: The training result from the wrapped strategy.
        """
        model_name = self.get_wrapped_model().__class__.__name__
        print(f"Starting training for {model_name}")
        result = super().train(X_train, y_train)
        print(f"Training completed for {model_name}")
        return result

    def predict(self, X_test) -> int:
        """
            Log the start of the prediction process and the prediction results.

            :return: The prediction result from the wrapped strategy.
        """
        model_name = self.get_wrapped_model().__class__.__name__
        print(f"Starting prediction for {model_name}")
        predictions = super().predict(X_test)
        print(f"Prediction result: {predictions}")
        return predictions

    def print_results(self, y_test, predictions):
        """
            Log the start and end of the result printing process.
        """
        print("Printing results...")
        super().print_results(y_test, predictions)
        print("Results printed.")