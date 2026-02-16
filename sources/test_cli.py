import numpy as np
from registry import registry

@registry.register(
    name='source.test_cli',
    type_='source',
    signature='measure(point: np.ndarray, n_samples: int) -> float'
)
class TestCLISource:
    def measure(self, point, n_samples=5):
        return float(np.sum(point**2) + np.random.normal(0, 0.1))
