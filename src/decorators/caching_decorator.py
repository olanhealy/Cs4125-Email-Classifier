# File: src/decorators/caching_decorator.py
from src.decorators.decorator import ClassifierDecorator
import hashlib
import pickle

class CachingDecorator(ClassifierDecorator):
    """
        A decorator to cache the predictions of the model.
        Avoids recomputing predictions for the same input data.
    """
    def __init__(self, strategy):
        """
            Initialise the caching decorator with an empty cache.
        """
        super().__init__(strategy)
        self.cache = {}

    def _hash_input(self, model, data):
        """
        Create a unique hash for the input data and model to use as the cache key.

        :returns: A hash string representing the input data.
        """
        # Create a unique identifier based on the model's class name
        # Create a unique identifier based on the model's class name and its memory address
        model_identifier = f"{model.__class__.__name__}_{id(model)}"

        # Create a hash of the input data
        data_hash = hashlib.md5(pickle.dumps(data)).hexdigest()

        # Combine model identifier and data hash to form the final cache key
        return f"{model_identifier}_{data_hash}"

    def train(self, X_train, y_train):
        """"
            Training process without caching
        """
        print("[DEBUG] Training with CachingDecorator...")
        return super().train(X_train, y_train)

    def predict(self, X_test):
        """
            Cache the predictions based on the hashed input data.
        """
        # Generate a hash of the input data
        data_hash = self._hash_input(self._strategy, X_test)

        # Check if result is already cached
        if data_hash in self.cache:
            print("[CACHE] Returning cached prediction.")
            return self.cache[data_hash]

        # If not cached, perform the prediction and cache it
        print("[CACHE] Caching new prediction...")
        predictions = super().predict(X_test)
        self.cache[data_hash] = predictions
        return predictions

    def print_results(self, y_test, predictions):
        """"
        Printing results without caching
        """
        print("[DEBUG] Printing results with CachingDecorator...")
        return super().print_results(y_test, predictions)
