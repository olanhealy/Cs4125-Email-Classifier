from src.preprocessing.preprocess import get_processed_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from utils import config

#THIS IS WHERE THE DATA IS CALLED AND IMPORTED FROM THE PREPROCESSING FUNCTIONS
train_data, test_data = get_processed_data()
max_features = config.get("preprocessing.max_features", 2000)
stop_words = config.get("preprocessing.stop_words", "english")
tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
X_train_vectorized = tfidf_vectorizer.fit_transform(train_data['Interaction content']).toarray()
X_test_vectorized = tfidf_vectorizer.transform(test_data['Interaction content']).toarray()

y_train = train_data['label']
y_test = test_data['label']

model = ExtraTreesClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)
print(classification_report(y_test, y_pred))