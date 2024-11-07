#src/decorators/logging_decorator.py
import logging
from src.models.base import BaseModel

class LoggingDecorator(BaseModel):
    def __init__(self, model):
        """
            Initialize the LoggingDecorator with a model to wrap.

            :param model: An instance of a classifier implementing BaseModel interface.
        """
        self.model = model

    def train(self, data):
        logging.info("Starting training...")
        result = self._model.train(data)
        logging.info("Training completed.")
        return result

    def predict(self, data):
        logging.info("Starting prediction...")
        result = self._model.predict(data)
        logging.info(f"Prediction result: {result}")
        return result

    def data_transform(self, data):
        """
        Pass-through for data transformation.
        """
        logging.info("Starting data transformation...")
        result = self._model.data_transform(data)
        logging.info("Data transformation completed.")
        return result

