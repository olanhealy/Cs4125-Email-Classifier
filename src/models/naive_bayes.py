from sklearn.naive_bayes import MultinomialNB

class NaiveBayesModel:
    """
    A wrapper for the Naive Bayes model.
    This model is particularly effective for text classification tasks and assumes independence between features.
    """
    def __init__(self, alpha=1.0):
        self.model = MultinomialNB(alpha=alpha)