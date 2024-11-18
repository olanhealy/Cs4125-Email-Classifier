class Observer:
    """Abstract observer class for the Observer Pattern."""
    def update(self, event_type: str, data: dict):
        raise NotImplementedError("Subclasses must implement this method.")


class Subject:
    """Subject class to manage observers and notify them."""
    def __init__(self):
        self._observers = []

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def notify_observers(self, event_type: str, data: dict):
        for observer in self._observers:
            observer.update(event_type, data)
