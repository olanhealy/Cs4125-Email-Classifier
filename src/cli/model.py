from src.preprocessing.preprocessor import preprocess_data_with_splits
from src.utils.config import Configuration
from src.utils.observer import Subject
from src.utils.logger import Logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pandas as pd
from src.models.model_factory import ClassifierFactory
from src.strategies.extra_trees_strategy import ExtraTreesStrategy
from src.strategies.gradient_boosting_strategy import GradientBoostingStrategy
from src.strategies.naive_bayes_strategy import NaiveBayesStrategy
from src.strategies.svm_strategy import SVMStrategy
from src.strategies.xgboost_strategy import XGBoostStrategy
from src.strategies.model_context import ModelContext

def run_model_with_strategy(dataset, model_name, result_format):
    # Initialize configuration
    config = Configuration()

    # Initialize subject and logger
    subject = Subject()
    logger = Logger()
    subject.add_observer(logger)

    # Notify observers about the process start
    subject.notify_observers("start", {"model": model_name, "csv": dataset})

    # Load and preprocess data
    input_csv = f"data/{dataset}"
    output_csv = f"data/preprocessed_{dataset}"
    df = pd.read_csv(input_csv)
    train_data, test_data = preprocess_data_with_splits(df, output_csv)

    # Vectorize data
    max_features = config.get("preprocessing.max_features", 2000)
    stop_words = config.get("preprocessing.stop_words", "english")
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    X_train_vectorized = tfidf_vectorizer.fit_transform(train_data['Interaction content']).toarray()
    X_test_vectorized = tfidf_vectorizer.transform(test_data['Interaction content']).toarray()
    y_train = train_data['label']
    y_test = test_data['label']

    # Use Factory to create model
    if model_name == "extra_trees":
        model_instance = ClassifierFactory.create_extra_trees_model()
        strategy = ExtraTreesStrategy(model_instance.model)
    elif model_name == "gradient_boosting":
        model_instance = ClassifierFactory.create_gradient_boosting_model()
        strategy = GradientBoostingStrategy(model_instance.model)
    elif model_name == "naive_bayes":
        model_instance = ClassifierFactory.create_naive_bayes_model()
        strategy = NaiveBayesStrategy(model_instance.model)
    elif model_name == "svm":
        model_instance = ClassifierFactory.create_svm_model()
        strategy = SVMStrategy(model_instance.model)
    elif model_name == "xg_boost":
        model_instance = ClassifierFactory.create_xgboost_model()
        strategy = XGBoostStrategy(model_instance.model)
    else:
        print(f"Invalid model name: {model_name}")
        return

    # Use ModelContext for training and evaluation
    context = ModelContext(strategy, format_type=result_format)

    # Notify observers about progress
    subject.notify_observers("progress", {"progress": 50})
    context.train(X_train_vectorized, y_train)
    subject.notify_observers("progress", {"progress": 75})

    # Predict and evaluate
    y_pred = context.predict(X_test_vectorized)
    subject.notify_observers("complete", {"results": classification_report(y_test, y_pred, output_dict=True)})

    # Print classification report
    context.print_results(y_test, y_pred)
