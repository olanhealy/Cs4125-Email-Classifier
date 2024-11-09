#src/decorators/base_decorator.py
from abc import ABC
from src.models.base import BaseModel

class BaseDecorator(BaseModel, ABC):
    """
        BaseDecorator class that wraps any model implementing the BaseModel interface.
        This class serves as the base for other decorators, enabling additional functionality
        to be added to models without modifying the original model code.
    """

    def __init__(self, base_model: BaseModel):
        """
            Initialize the BaseDecorator with the model to be wrapped.

            :param base_model: An instance of a classifier implementing the BaseModel interface.
        """
        super().__init__()
        self.base_model = base_model

    def train(self) -> None:
        """
            This method calls the train() method of the wrapped model.
        """
        self.base_model.train()

    def predict(self) -> int:
        """
            This method calls the predict() method of the wrapped model and returns its result.

            :return: The prediction result from the wrapped model.
        """
        return self.base_model.predict()

    def data_transform(self) -> None:
        """
            This method calls the data_transform() method of the wrapped model.
        """
        return self.base_model.data_transform()
