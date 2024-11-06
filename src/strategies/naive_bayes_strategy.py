from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from strategies.classification_strategy import ClassificationStrategy

class NaiveBayesStrategy(ClassificationStrategy):
    def __init__(self, alpha=1.0):
        self.model = MultinomialNB(alpha=alpha)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def print_results(self, y_test, predictions):
        print(classification_report(y_test, predictions))