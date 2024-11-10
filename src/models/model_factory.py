from models.gradient_boosting import GradientBoostingModel
from models.naive_bayes import NaiveBayesModel
from models.xg_boost import XGBoostModel
from models.svm import SVMModel
from models.extra_trees import ExtraTreesModel

class ClassifierFactory:
    """
    Factory class for creating different types of classifiers.
    This encapsulates the logic for creating various classifier models.
    """

    @staticmethod
    def create_model(model_class, **kwargs):
        return model_class(**kwargs)

    @classmethod
    def create_naive_bayes_model(cls, alpha=1.0):
        return cls.create_model(NaiveBayesModel, alpha=alpha)

    @classmethod
    def create_svm_model(cls, kernel='linear', C=1.0, random_state=0):
        return cls.create_model(SVMModel, kernel=kernel, C=C, random_state=random_state)

    @classmethod
    def create_gradient_boosting_model(cls, n_estimators=100, learning_rate=0.1, random_state=0):
        return cls.create_model(GradientBoostingModel, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    @classmethod
    def create_xgboost_model(cls, n_estimators=100, learning_rate=0.1, random_state=0):
        return cls.create_model(XGBoostModel, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    @classmethod
    def create_extra_trees_model(cls, n_estimators=1000, random_state=0, class_weight='balanced'):
        return cls.create_model(ExtraTreesModel, n_estimators=n_estimators, random_state=random_state, class_weight=class_weight)