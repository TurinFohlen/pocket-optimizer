import numpy as np
from registry import registry


@registry.register(
    name='source.test_function',
    type_='source',
    signature='measure(point: np.ndarray, n_samples: int) -> float'
)
class TestFunctionSource:
    def __init__(self):
        self.measurement_count = 0
        self.noise_level = 0.01
    
    def measure(self, point: np.ndarray, n_samples: int = 5) -> float:
        self.measurement_count += 1
        
        true_value = self._complex_test_function(point)
        
        measurements = []
        for _ in range(n_samples):
            noise = np.random.normal(0, self.noise_level)
            measurements.append(true_value + noise)
        
        return float(np.mean(measurements))
    
    def _complex_test_function(self, point: np.ndarray) -> float:
        point = np.atleast_1d(point)
        
        if len(point) == 2:
            x, y = point[0], point[1]
            z = 0.5
        elif len(point) >= 3:
            x, y, z = point[0], point[1], point[2]
        else:
            x = point[0]
            y = 0.5
            z = 0.5
        
        p1 = np.array([0.3, 0.3, 0.3])
        p2 = np.array([0.7, 0.7, 0.7])
        
        d1 = (x - p1[0])**2 + (y - p1[1])**2 + (z - p1[2])**2
        d2 = (x - p2[0])**2 + (y - p2[1])**2 + (z - p2[2])**2 + 0.05
        
        f = np.minimum(d1, d2)
        
        noise = 0.015 * (np.sin(20 * x) * np.sin(25 * y) * np.sin(30 * z))
        f += noise
        
        f = np.where(f > 0.2, 0.2 + 0.1 * np.log1p(f - 0.2), f)
        
        return -float(f)
