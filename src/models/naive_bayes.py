import numpy as np
import pandas as pd
from models.base import BaseModel
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

class NaiveBayesModel(BaseModel):
    def __init__(self, model_name: str, alpha: float = 1.0, **kwargs) -> None:
        super(NaiveBayesModel, self).__init__()
        self.model_name = model_name
        self.mdl = MultinomialNB(alpha=alpha)
        self.predictions = None

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        self.predictions = self.mdl.predict(X_test)

    def print_results(self, data):
        print(classification_report(data.y_test, self.predictions))
