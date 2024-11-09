#src/decorators/error_handling_decorator.py
from src.decorators.base_decorator import BaseDecorator
from src.models.base import BaseModel

class ErrorHandlingDecorator(BaseDecorator):
    """
    A decorator class that adds error handling functionality to any model implementing the BaseModel interface.
    This decorator logs any exceptions that occur during training and prediction.
    """

    def __init__(self, base_model: BaseModel):
        """
        Initialize the ErrorHandlingDecorator with the model to be wrapped.

        :param base_model: An instance of a classifier implementing the BaseModel interface.
        """
        super().__init__(base_model)

    def train(self) -> None:
        """
        Wrap the train() method to catch and log any exceptions that occur.
        If an exception occurs, it logs the error but re-raises it.
        """
        try:
            super().train()
        except Exception as e:
            model_name = self.get_wrapped_model().__class__.__name__
            self.logger.error(f"Error during training for {model_name}: {e}")
            raise  # Re-raise the exception to let other decorators handle it

    def predict(self) -> int:
        """
        Wrap the predict() method to catch and log any exceptions that occur.
        If an exception occurs, it logs the error but re-raises it.

        :return: The prediction result from the wrapped model, or -1 if an error occurs.
        """
        try:
            return super().predict()
        except Exception as e:
            model_name = self.get_wrapped_model().__class__.__name__
            self.logger.error(f"Error during prediction for {model_name}: {e}")
            raise  # Re-raise the exception to let other decorators handle it
