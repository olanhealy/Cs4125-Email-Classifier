import numpy as np
import pandas as pd
from models.base import BaseModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report

class ExtraTreesModel(BaseModel):
    def __init__(self, model_name: str, n_estimators: int, random_state: int, class_weight: str, **kwargs) -> None:
        super(ExtraTreesModel, self).__init__()
        self.model_name = model_name
        self.mdl = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state, class_weight=class_weight)
        self.predictions = None

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        self.predictions = self.mdl.predict(X_test)

    def print_results(self, data):
        print(classification_report(data.y_test, self.predictions))
