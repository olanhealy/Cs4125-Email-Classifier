from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from strategies.classification_strategy import ClassificationStrategy

class XGBoostStrategy(ClassificationStrategy):
    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=0):
        self.model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def print_results(self, y_test, predictions):
        print(classification_report(y_test, predictions))