from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from strategies.classification_strategy import ClassificationStrategy

class ExtraTreesStrategy(ClassificationStrategy):
    def __init__(self, n_estimators=1000, random_state=0, class_weight='balanced'):
        self.model = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state, class_weight=class_weight)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def print_results(self, y_test, predictions):
        print(classification_report(y_test, predictions))
