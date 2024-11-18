import pandas as pd
from src.preprocessing.preprocessor import preprocess_data_with_splits
from src.utils.config import Configuration
from src.utils.observer import Subject
from src.utils.logger import Logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report

# Initialize configuration
config = Configuration()

# Initialize subject and logger
subject = Subject()
logger = Logger()
subject.add_observer(logger)

# Notify observers that the process has started
subject.notify_observers("start", {"model": "ExtraTreesClassifier", "csv": "data/AppGallery.csv"})

# Load and preprocess data
input_csv = 'data/Purchasing.csv'
output_csv = 'data/preprocessed_Purchasing_data_with_types.csv'
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

# Train model
model = ExtraTreesClassifier(n_estimators=100, random_state=42)
subject.notify_observers("progress", {"progress": 50})
model.fit(X_train_vectorized, y_train)
subject.notify_observers("progress", {"progress": 75})

# Predict and notify completion
y_pred = model.predict(X_test_vectorized)
classification_results = classification_report(y_test, y_pred, output_dict=True)
subject.notify_observers("complete", {"results": classification_results})

# Print classification report
print(classification_report(y_test, y_pred))
