from .base_generator import BaseGenerator
import numpy as np

class Generator(BaseGenerator):
    """
    This is generator utilizes gamma distribution to generate load correlated vectors with gamma distribution
    ref: https://en.wikipedia.org/wiki/Gamma_distribution
    """
    
    def __init__(self) -> None:
        # TODO
        pass

    def get_estimated_cloud_load(self) -> float:
        # TODO
        pass

    def generate_cloud_load_vectors(self) -> np.ndarray:
        # TODO
        pass


if __name__ == '__main__':
    pass