from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from strategies.classification_strategy import ClassificationStrategy

class GradientBoostingStrategy(ClassificationStrategy):
    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=0):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def print_results(self, y_test, predictions):
        print(classification_report(y_test, predictions))