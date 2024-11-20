from src.models.gradient_boosting import GradientBoostingModel
from src.models.naive_bayes import NaiveBayesModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.svm import SVMModel
from src.models.extra_trees import ExtraTreesModel
from src.utils.config import Configuration

class ClassifierFactory:
    """
    Factory class for creating different types of classifiers.
    This encapsulates the logic for creating various machine learning models dynamically.
    """

    @staticmethod
    def create_model(model_class, **kwargs):
        """
        Static method for creating a model instance.

        """
        return model_class(**kwargs)

    @classmethod
    def create_naive_bayes_model(cls):
        """
        Create and return a Naive Bayes model using parameters from the configuration.

        :return: An instance of NaiveBayesModel.
        """
        config = Configuration()
        alpha = config.get("model_params.naive_bayes.alpha", 1.0)
        return cls.create_model(NaiveBayesModel, alpha=alpha)

    @classmethod
    def create_svm_model(cls):
        """
        Create and return an SVM model using parameters from the configuration.

        :return: An instance of SVMModel.
        """
        config = Configuration()
        C = config.get("model_params.svm.C", 1.0)
        kernel = config.get("model_params.svm.kernel", 'linear')
        random_state = config.get("model_params.svm.random_state", 0)
        return cls.create_model(SVMModel, kernel=kernel, C=C, random_state=random_state)

    @classmethod
    def create_gradient_boosting_model(cls):
        """
        Create and return a Gradient Boosting model using parameters from the configuration.

        :return: An instance of GradientBoostingModel.
        """
        config = Configuration()
        n_estimators = config.get("model_params.gradient_boosting.n_estimators", 100)
        learning_rate = config.get("model_params.gradient_boosting.learning_rate", 0.1)
        random_state = config.get("model_params.gradient_boosting.random_state", 0)
        return cls.create_model(GradientBoostingModel, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    @classmethod
    def create_logistic_regression_model(cls):
        """
        Create and return a Logistic Regression model using parameters from the configuration.

        :return: An instance of LogisticRegressionModel.
        """
        config = Configuration()
        penalty = config.get("model_params.logistic_regression.penalty", "l2")
        C = config.get("model_params.logistic_regression.C", 1.0)
        random_state = config.get("model_params.logistic_regression.random_state", 0)
        return cls.create_model(LogisticRegressionModel, penalty=penalty, C=C, random_state=random_state)


    @classmethod
    def create_extra_trees_model(cls):
        """
        Create and return an Extra Trees model using parameters from the configuration.

        :return: An instance of ExtraTreesModel.
        """
        config = Configuration()
        n_estimators = config.get("model_params.extra_trees.n_estimators", 1000)
        random_state = config.get("model_params.extra_trees.random_state", 0)
        class_weight = config.get("model_params.extra_trees.class_weight", 'balanced')
        return cls.create_model(ExtraTreesModel, n_estimators=n_estimators, random_state=random_state, class_weight=class_weight)