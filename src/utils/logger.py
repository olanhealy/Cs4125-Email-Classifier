from src.utils.observer import Observer

class Logger(Observer):
    """Logger that stores events and prints a summary at the end."""
    def __init__(self):
        self.logs = []

    def update(self, event_type: str, data: dict):
        if event_type == "start":
            self.logs.append(f"Process started with model: {data['model']} and CSV: {data['csv']}")
        elif event_type == "progress":
            self.logs.append(f"Progress: {data['progress']}%")
        elif event_type == "complete":
            self.logs.append(f"Process completed. Results: {data['results']}")
            self.print_logs()

    def print_logs(self):
        print("\n[LOGGER] Summary of Events:")
        for log in self.logs:
            print(log)
