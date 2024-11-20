from sklearn.ensemble import ExtraTreesClassifier

class ExtraTreesModel:

    """
    A wrapper for the Extra Trees Classifier.
    This model is part of the ensemble learning methods and builds multiple decision trees.
    """
    
    def __init__(self, n_estimators=1000, random_state=0, class_weight='balanced'):
        self.model = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state, class_weight=class_weight)
