from sklearn.ensemble import GradientBoostingClassifier

class GradientBoostingModel:

    """
    A wrapper for the Gradient Boosting Classifier.
    This model is used for sequentially improving predictions by correcting previous errors.
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=0):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
