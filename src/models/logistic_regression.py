from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    
    """
    A wrapper for the Logistic Regression model.
    A simple yet effective linear model for classification tasks.
    """

    def __init__(self, penalty="l2", C=1.0, random_state=0):
        self.model = LogisticRegression(penalty=penalty, C=C, random_state=random_state)