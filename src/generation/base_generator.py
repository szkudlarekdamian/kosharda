from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    """
    Base class for instances generator
    """
    
    @abstractmethod
    def get_estimated_cloud_load(self) -> float:
        pass

    @abstractmethod
    def generate_cloud_load_vectors(self):
        pass


if __name__ == '__main__':
    pass