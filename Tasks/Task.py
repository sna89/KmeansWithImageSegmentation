from abc import ABC, abstractmethod

class AbstractTask(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def _preprocess_data(self):
        pass

    @abstractmethod
    def _fit(self):
        pass

    @abstractmethod
    def execute(self):
        pass
