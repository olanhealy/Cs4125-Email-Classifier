from src.utils.observer import Observer

class Logger(Observer):
    """Logger that stores events and prints a summary at the end."""
    def __init__(self):
        self.logs = []

    def update(self, event_type: str, data: dict):
        if event_type == "start":
            message = f"Process started with model: {data['model']} and CSV: {data['csv']}"
        elif event_type == "progress":
            message = f"Progress: {data['progress']}% - {data.get('message', 'Action in progress')}"
        elif event_type == "complete":
            message = f"Process completed: {data['message']}"
        else:
            message = f"Unknown event type: {event_type}"

        # Log and print the message
        self.logs.append(message)
        print(f"[LOGGER] {message}")

        # If complete, print the summary of all events
        if event_type == "complete":
            self.print_logs()

    def print_logs(self):
        print("\n[LOGGER] Summary of Events:")
        for log in self.logs:
            print(log)
