from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from src.preprocessing.preprocessor import preprocess_data_with_splits
from src.utils.config import Configuration
from src.utils.observer import Subject
from src.utils.logger import Logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd

def run_model(dataset, model_name):
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
    max_features = config.get_setting("preprocessing.max_features", 2000)
    stop_words = config.get_setting("preprocessing.stop_words", "english")
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    X_train_vectorized = tfidf_vectorizer.fit_transform(train_data['Interaction content']).toarray()
    X_test_vectorized = tfidf_vectorizer.transform(test_data['Interaction content']).toarray()
    y_train = train_data['label']
    y_test = test_data['label']

    # Initialize LabelEncoder for XGBoost
    label_encoder = None
    if model_name == "xg_boost":
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
    else:
        y_test_encoded = y_test  # Keep original for non-XGBoost models

    # Select model
    if model_name == "extra_trees":
        model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    elif model_name == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_name == "naive_bayes":
        model = MultinomialNB()
    elif model_name == "svm":
        model = SVC(kernel='linear', random_state=42)
    elif model_name == "xg_boost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        print(f"Invalid model name: {model_name}")
        return

    # Notify observers about progress
    subject.notify_observers("progress", {"progress": 50})
    model.fit(X_train_vectorized, y_train)
    subject.notify_observers("progress", {"progress": 75})

    # Predict and decode labels for XGBoost
    y_pred = model.predict(X_test_vectorized)
    if model_name == "xg_boost":
        y_pred = label_encoder.inverse_transform(y_pred)

    # Notify observers about completion
    classification_results = classification_report(y_test_encoded, y_pred, output_dict=True)
    subject.notify_observers("complete", {"results": classification_results})

    # Print classification report
    print(classification_report(y_test_encoded, y_pred))
