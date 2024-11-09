#src/decorators/logging_decorator.py
import logging
from src.decorators.base_decorator import BaseDecorator
from src.models.base import BaseModel

# Configure logging to print to console
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

class LoggingDecorator(BaseDecorator):
    """
    A decorator class that adds logging functionality to any model implementing the BaseModel interface.
    This decorator logs when training starts and completes, as well as when predictions are made.
    """

    def __init__(self, base_model: BaseModel):
        """
            Initialise the LoggingDecorator with a model to wrap.

            :param base_model: An instance of a classifier implementing BaseModel interface.
        """
        super().__init__(base_model)

    def train(self) -> None:
        """
            Log the start and completion of the training process.
            This method calls the train() method of the wrapped model and logs messages before and after the call.
        """
        logging.info(f"Starting training for {self.base_model.__class__.__name__}")
        super().train()
        logging.info(f"Training completed for {self.base_model.__class__.__name__}")

    def predict(self) -> int:
        """
            Log the start of the prediction process and the prediction results.
            This method calls the predict() method of the wrapped model and logs messages before and after the call.

            :return: The prediction result from the wrapped model.
        """
        logging.info(f"Starting prediction for {self.base_model.__class__.__name__}")
        result = super().predict()
        logging.info(f"Prediction result: {result}")
        return result