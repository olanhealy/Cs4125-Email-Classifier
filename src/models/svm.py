from sklearn.svm import SVC

class SVMModel:
    def __init__(self, kernel='linear', C=1.0, random_state=0):
        self.model = SVC(kernel=kernel, C=C, probability=True, random_state=random_state)