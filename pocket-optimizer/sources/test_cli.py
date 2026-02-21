import numpy as np
from registry import registry

@registry.register(
    name='source.test_cli',
    type_='source',
    signature='measure(point: np.ndarray) -> float'
)
class TestCLISource:
    def __init__(self, n_samples: int = 5):
        self.n_samples = n_samples

    def measure(self, point) -> float:
        return float(np.sum(point**2) + np.random.normal(0, 0.1))
