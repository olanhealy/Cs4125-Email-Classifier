#src/decorators/logging_decorator.py
from src.decorators.base_decorator import BaseDecorator
from src.models.base import BaseModel

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
        model_name = self.get_wrapped_model().__class__.__name__
        self.logger.info(f"Starting training for {model_name}")
        super().train()
        self.logger.info(f"Training completed for {model_name}")

    def predict(self) -> int:
        """
            Log the start of the prediction process and the prediction results.
            This method calls the predict() method of the wrapped model and logs messages before and after the call.

            :return: The prediction result from the wrapped model.
        """
        model_name = self.get_wrapped_model().__class__.__name__
        self.logger.info(f"Starting prediction for {model_name}")
        result = super().predict()
        self.logger.info(f"Prediction result: {result}")
        return result