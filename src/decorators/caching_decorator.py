#src/decorators/caching_decorator.py
from src.decorators.base_decorator import BaseDecorator
from src.models.base import BaseModel
import numpy as np


class CachingDecorator(BaseDecorator):
    """
    A decorator class that caches the results of predictions for faster repeated predictions.
    """

    def __init__(self, base_model: BaseModel):
        super().__init__(base_model)

    def predict(self, X=None) -> int:
        """
        Override the predict method to add caching functionality.
        Note: We use a workaround to keep the existing `predict()` method signature.
        """
        if X is None:
            return super().predict()  # Use the existing predict method if no input is provided

        # Convert X to a hashable key
        cache_key = X.tobytes() if isinstance(X, np.ndarray) else str(X)

        # Check if the result is already cached
        if cache_key in self.cache:
            self.logger.info("Cache hit for input")
            return self.cache[cache_key]

        # Cache miss: call the original predict and store the result
        self.logger.info("Cache miss for input. Computing prediction.")
        self.base_model.X_test = X  # Set the test data on the model
        result = super().predict()
        self.cache[cache_key] = result
        return result
