from src.strategies.classification_strategy import ClassificationStrategy
from src.decorators.error_handling_decorator import ErrorHandlingDecorator
from src.decorators.logging_decorator import LoggingDecorator
from src.decorators.result_formatting_decorator import ResultFormattingDecorator
from src.decorators.timing_decorator import TimingDecorator

class ModelContext:
    """
    Context class for managing classification strategies.
    Dynamically applies decorators for additional functionality such as logging, error handling, and formatting.
    """
    def __init__(self, strategy: ClassificationStrategy,
        use_logging=True,
        use_timing=True,
        use_error_handling=True,
        use_formatting=True,
        format_type='text'):

        # Apply the decorators based on the provided flags
        if use_formatting:
            strategy = ResultFormattingDecorator(strategy, format_type=format_type)
        """
        Initialise the context with a chosen strategy and optional decorators.

        :param strategy: An instance of a ClassificationStrategy.
        :param use_logging: Flag to enable logging.
        :param use_timing: Flag to enable timing.
        :param use_error_handling: Flag to enable error handling.
        :param use_formatting: Flag to enable result formatting.
        :param format_type: Desired format for results (e.g., text, JSON, table).
        """


        if use_logging:
            strategy = LoggingDecorator(strategy)
        if use_timing:
            strategy = TimingDecorator(strategy)
        if use_error_handling:
            strategy = ErrorHandlingDecorator(strategy)

        self.strategy = strategy

    def train(self, X_train, y_train):
        """
        Train the model using the selected strategy.
        """
        self.strategy.train(X_train, y_train)

    def predict(self, X_test):
        """
        Predict labels using the selected strategy.

        :return: Predicted labels.
        """
        return self.strategy.predict(X_test)

    def print_results(self, y_test, predictions):
        """
        Print the classification results using the selected strategy.
        """
        self.strategy.print_results(y_test, predictions)