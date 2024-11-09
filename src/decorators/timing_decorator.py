#src/decorators/timing_decorator.py
import time
import logging
from src.decorators.base_decorator import BaseDecorator
from src.models.base import BaseModel

# Configure logging to print to console
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

class TimingDecorator(BaseDecorator):
    """
        A decorator class that adds timing functionality to any model implementing the BaseModel interface.
        This decorator logs the time taken for training and prediction.
    """

    def __init__(self, base_model: BaseModel):
        """
            Initialize the TimingDecorator with the model to be wrapped.

            :param base_model: An instance of a classifier implementing the BaseModel interface.
        """
        super().__init__(base_model)

    def train(self) -> None:
        """
            Measure and log the time taken to train the model.
            This method wraps the train() method of the wrapped model and logs the elapsed time.
        """
        start_time = time.time()
        super().train()
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Training took {elapsed_time:.4f} seconds")

    def predict(self) -> int:
        """
            Measure and log the time taken to make a prediction.
            This method wraps the predict() method of the wrapped model and logs the elapsed time.

            :return: The prediction result from the wrapped model.
        """
        start_time = time.time()
        result = super().predict()
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Prediction took {elapsed_time:.4f} seconds")
        return result
