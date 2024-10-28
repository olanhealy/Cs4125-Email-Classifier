# src/classifier_factory.py

from models.extra_trees import ExtraTreesModel
from models.gradient_boosting import GradientBoostingModel
from models.naive_bayes import NaiveBayesModel
from models.xg_boost import XGBoostModel
from models.svm import SVMModel

class ClassifierFactory:
    """
    Factory class for creating different types of classifiers.
    This encapsulates the logic for creating various classifier models.
    """

    @staticmethod
    def create_model(model_class, model_name, **kwargs):
        return model_class(model_name=model_name, **kwargs)

    @classmethod
    def create_extra_trees_model(cls, model_name, n_estimators=1000, random_state=0, class_weight='balanced', **kwargs):
        return cls.create_model(ExtraTreesModel, model_name, n_estimators=n_estimators, random_state=random_state, class_weight=class_weight, **kwargs)

    @classmethod
    def create_gradient_boosting_model(cls, model_name, n_estimators=100, learning_rate=0.1, random_state=0, **kwargs):
        return cls.create_model(GradientBoostingModel, model_name, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state, **kwargs)

    @classmethod
    def create_xgboost_model(cls, model_name, n_estimators=100, learning_rate=0.1, random_state=0, use_label_encoder=False, eval_metric='mlogloss', **kwargs):
        return cls.create_model(XGBoostModel, model_name, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state, use_label_encoder=use_label_encoder, eval_metric=eval_metric, **kwargs)
    
    @classmethod
    def create_svm_model(cls, model_name, kernel='linear', C=1.0, random_state=0, **kwargs):
        return cls.create_model(SVMModel, model_name, kernel=kernel, C=C, random_state=random_state, **kwargs)
    
    @classmethod
    def create_naive_bayes_model(cls, model_name, alpha=1.0, **kwargs):
        return cls.create_model(NaiveBayesModel, model_name, alpha=alpha, **kwargs)
