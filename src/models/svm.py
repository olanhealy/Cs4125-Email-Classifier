import numpy as np
import pandas as pd
from models.base import BaseModel
from sklearn.svm import SVC
from sklearn.metrics import classification_report

class SVMModel(BaseModel):
    def __init__(self, model_name: str, kernel: str = 'linear', C: float = 1.0, random_state: int = 0, **kwargs) -> None:
        super(SVMModel, self).__init__()
        self.model_name = model_name
        self.mdl = SVC(kernel=kernel, C=C, probability=True, random_state=random_state)
        self.predictions = None

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        self.predictions = self.mdl.predict(X_test)

    def print_results(self, data):
        print(classification_report(data.y_test, self.predictions))
