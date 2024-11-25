import os
import json
from tabulate import tabulate  
import warnings
from sklearn.exceptions import UndefinedMetricWarning

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
from src.strategies.logistic_regression_strategy import LogisticRegressionStrategy
from src.strategies.model_context import ModelContext

"""

This script handles the core workflow of the Email Classifier application, including:
1. Loading and preprocessing the dataset.
2. Configuring and initialising the chosen machine learning model via the Factory Pattern.
3. Applying the Strategy Pattern to encapsulate model-specific behavior.
4. Training the model, making predictions, and evaluating results.
5. Logging progress and saving results in the specified format (text, JSON, or table).

The script uses pur design patterns:
- Factory Pattern: Dynamically creates model instances.
- Strategy Pattern: Encapsulates different classification strategies.
- Observer Pattern: Logs progress and completion using a subject-observer relationship.

The `helper` function is called from the CLI and performs all operations seamlessly.

Inputs:
    - Dataset file (CSV)
    - Selected model name
    - Result format

Outputs:
    - Processed results saved in the `results/` directory.
"""

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import os
import json
from tabulate import tabulate
import warnings
from sklearn.exceptions import UndefinedMetricWarning

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
from src.strategies.logistic_regression_strategy import LogisticRegressionStrategy
from src.strategies.model_context import ModelContext

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def helper(dataset, model_name, result_format):
    # Initialise configuration
    config = Configuration()

    # Initialise subject and logger
    subject = Subject()
    logger = Logger()
    subject.add_observer(logger)

    # Notify observers about the process start
    subject.notify_observers("start", {"model": model_name, "csv": dataset})

    # Step 1: Load dataset
    input_csv = f"data/{dataset}"
    df = pd.read_csv(input_csv)
    subject.notify_observers("progress", {"progress": 10, "message": "Dataset loaded successfully."})

    # Step 2: Preprocess data
    train_data, test_data = preprocess_data_with_splits(df)
    subject.notify_observers("progress", {"progress": 25, "message": "Data preprocessed and split into training and testing sets."})

    # Step 3: Vectorise data
    max_features = config.get("preprocessing.max_features", 2000)
    stop_words = config.get("preprocessing.stop_words", "english")
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    X_train_vectorized = tfidf_vectorizer.fit_transform(train_data['Interaction content']).toarray()
    X_test_vectorized = tfidf_vectorizer.transform(test_data['Interaction content']).toarray()
    y_train = train_data['label']
    y_test = test_data['label']
    subject.notify_observers("progress", {"progress": 40, "message": "Data vectorized using TF-IDF."})

    # Step 4: Use Factory to create model
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
    elif model_name == "logistic_regression":
        model_instance = ClassifierFactory.create_logistic_regression_model()
        strategy = LogisticRegressionStrategy(model_instance.model)
    else:
        subject.notify_observers("error", {"message": f"Invalid model name: {model_name}"})
        return
    subject.notify_observers("progress", {"progress": 50, "message": f"Model '{model_name}' created using the Factory Pattern."})

    # Step 5: Use ModelContext for training
    context = ModelContext(strategy, format_type=result_format)
    context.train(X_train_vectorized, y_train)
    subject.notify_observers("progress", {"progress": 65, "message": "Model training completed."})

    # Step 6: Predict and evaluate
    y_pred = context.predict(X_test_vectorized)
    classification_data = classification_report(y_test, y_pred, output_dict=True)
    subject.notify_observers("progress", {"progress": 80, "message": "Model predictions completed."})

    # Step 7: Save results to `results` directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    file_name = os.path.join(results_dir, f"classification_{model_name}.{result_format}")
    if result_format == "json":
        with open(file_name, "w") as file:
            json.dump(classification_data, file, indent=4)
    elif result_format == "text":
        with open(file_name, "w") as file:
            file.write(classification_report(y_test, y_pred))
    elif result_format == "table":
        # Format the classification report as a table
        table_data = [
            [label] + list(metrics.values())
            for label, metrics in classification_data.items()
            if isinstance(metrics, dict)
        ]
        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]

        # Convert to tabular format
        table_output = tabulate(table_data, headers=headers, tablefmt="grid")

        # Save table to .txt file
        with open(file_name, "w") as file:
            file.write(table_output)
    subject.notify_observers("progress", {"progress": 90, "message": f"Results saved in '{file_name}'."})

    # Step 8: Print classification report
    context.print_results(y_test, y_pred)
    subject.notify_observers("complete", {
        "progress": 100,
        "message": "Process completed successfully.",
    })


