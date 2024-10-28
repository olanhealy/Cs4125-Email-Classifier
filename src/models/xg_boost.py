import numpy as np
import pandas as pd
from models.base import BaseModel
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

class XGBoostModel(BaseModel):
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        super(XGBoostModel, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', random_state=0)
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
