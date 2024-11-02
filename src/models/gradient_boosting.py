import numpy as np
import pandas as pd
from models.base import BaseModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

class GradientBoostingModel(BaseModel):
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        super(GradientBoostingModel, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0)
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        self.predictions = self.mdl.predict(X_test)

    def print_results(self, data):
        print(classification_report(data.y_test, self.predictions))

    def data_transform(self) -> None:
        pass  # Placeholder for any specific data transformation
