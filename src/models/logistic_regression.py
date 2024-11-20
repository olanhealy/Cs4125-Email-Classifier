from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self, penalty="l2", C=1.0, random_state=0):
        self.model = LogisticRegression(penalty=penalty, C=C, random_state=random_state)