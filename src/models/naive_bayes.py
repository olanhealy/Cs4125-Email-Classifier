from sklearn.naive_bayes import MultinomialNB

class NaiveBayesModel:
    def __init__(self, alpha=1.0):
        self.model = MultinomialNB(alpha=alpha)