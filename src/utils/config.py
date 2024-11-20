import json
import os

class Configuration:
    """
    Singleton class for managing application configuration.
    Ensures a single, globally accessible instance that reads settings from a configuration file (config.json).
    """
    _instance = None

    def __new__(cls):
        """
        Override the __new__ method to ensure only one instance of Configuration exists.
        If an instance doesn't exist, it creates one and initializes it.

        :return: Singleton instance of Configuration.
        """
        if cls._instance is None:
            cls._instance = super(Configuration, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """
        Load configuration settings from a JSON file (config.json).
        If the file is not found, initializes with an empty dictionary and logs an error.

        :raises: FileNotFoundError if the configuration file is missing.
        """
        config_path = os.path.join(os.path.dirname(__file__), '../../config.json')
        try:
            with open(config_path, 'r') as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_path}")
            self.settings = {}

    def get(self, key, default=None):
        """
        Retrieve a value from the configuration using a dot-separated key.
        Supports nested access to dictionary values.

        :param key: A string representing the configuration key (e.g., "model_params.svm.C").
        :param default: A default value to return if the key is not found.
        :return: The value associated with the key, or the default value if the key is not found.
        """
        keys = key.split('.')
        value = self.settings
        for k in keys:
            value = value.get(k, default)
            if value is default:
                break
        return value