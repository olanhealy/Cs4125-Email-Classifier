from src.strategies.classification_strategy import ClassificationStrategy
from src.decorators.error_handling_decorator import ErrorHandlingDecorator
from src.decorators.logging_decorator import LoggingDecorator
from src.decorators.result_formatting_decorator import ResultFormattingDecorator
from src.decorators.timing_decorator import TimingDecorator

class ModelContext:
    def __init__(self, strategy: ClassificationStrategy,
        use_logging=True,
        use_timing=True,
        use_error_handling=True,
        use_formatting=True,
        format_type='text'):

        # Apply the decorators based on the provided flags
        if use_formatting:
            strategy = ResultFormattingDecorator(strategy, format_type=format_type)
        if use_logging:
            strategy = LoggingDecorator(strategy)
        if use_timing:
            strategy = TimingDecorator(strategy)
        if use_error_handling:
            strategy = ErrorHandlingDecorator(strategy)

        self.strategy = strategy

    def train(self, X_train, y_train):
        self.strategy.train(X_train, y_train)

    def predict(self, X_test):
        return self.strategy.predict(X_test)

    def print_results(self, y_test, predictions):
        self.strategy.print_results(y_test, predictions)