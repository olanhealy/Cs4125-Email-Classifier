from sklearn.svm import SVC
from sklearn.metrics import classification_report
from strategies.classification_strategy import ClassificationStrategy

class SVMStrategy(ClassificationStrategy):
    def __init__(self, kernel='linear', C=1.0, random_state=0):
        self.model = SVC(kernel=kernel, C=C, probability=True, random_state=random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def print_results(self, y_test, predictions):
        print(classification_report(y_test, predictions))
