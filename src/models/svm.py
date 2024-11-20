from sklearn.svm import SVC

class SVMModel:
    """
    A wrapper for the Support Vector Machine (SVM) model.
    SVM is a powerful classification algorithm that works well with smaller datasets and handles high-dimensional spaces.
    """
    def __init__(self, kernel='linear', C=1.0, random_state=0):
        self.model = SVC(kernel=kernel, C=C, probability=True, random_state=random_state)