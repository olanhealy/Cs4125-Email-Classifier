import json
import os

class Configuration:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Configuration, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # Load config.json from the root of the project
        config_path = os.path.join(os.path.dirname(__file__), '../../config.json')
        try:
            with open(config_path, 'r') as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_path}")
            self.settings = {}

    def get_setting(self, key, default=None):
        # Nested access to configuration using dot notation
        keys = key.split('.')
        value = self.settings
        for k in keys:
            value = value.get(k, default)
            if value is default:
                break
        return value