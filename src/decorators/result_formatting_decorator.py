# src/decorators/result_formatting_decorator.py
import json
from tabulate import tabulate
from sklearn.metrics import classification_report
from src.decorators.decorator import ClassifierDecorator


class ResultFormattingDecorator(ClassifierDecorator):
    """
    A decorator to format the output of the print_results method.
    Supports plain text, JSON, and table formats.
    """

    def __init__(self, strategy, format_type='text'):
        """
        Initialise the decorator with the wrapped strategy and the desired format type.
        """
        super().__init__(strategy)
        self.format_type = format_type

    def print_results(self, y_test, predictions):
        """
        Format the classification results based on the chosen format type.
        """
        # Generate the classification report as a dictionary
        report = classification_report(y_test, predictions, output_dict=True)

        if self.format_type == 'json':
            print(self.format_json(report))
        elif self.format_type == 'table':
            print(self.format_table(report))
        else:
            print(self.format_text(y_test, predictions))

    def format_json(self, report):
        """
        Format the results as a JSON string.
        """
        # Add support for accuracy using the support from 'weighted avg'
        total_support = report['weighted avg']['support']
        report['accuracy'] = {
            "value": report['accuracy'],
            "support": total_support
        }
        return json.dumps(report, indent=4)

    def format_table(self, report):
        """
        Format the results as a table using the tabulate library.
        """
        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        table = []

        # Get the total support from the 'weighted avg' row, which represents the total sample size
        total_support = report['weighted avg']['support']

        # Loop through the report items
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                table.append([
                    label,
                    metrics.get('precision', ""),
                    metrics.get('recall', ""),
                    metrics.get('f1-score', ""),
                    metrics.get('support', "")
                ])
            elif label == "accuracy":
                table.append(["Accuracy", "", "", f"{report['accuracy']:.2f}", total_support])

        return tabulate(table, headers, tablefmt="grid")

    def format_text(self, y_test, predictions):
        """
        Format the results as plain text.
        """
        return classification_report(y_test, predictions)
