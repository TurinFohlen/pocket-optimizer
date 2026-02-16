import numpy as np
from registry import registry


@registry.register(
    name='source.interactive',
    type_='source',
    signature='measure(point: np.ndarray, n_samples: int) -> float'
)
class InteractiveSource:
    def __init__(self):
        self.measurement_count = 0
        self.history = []
    
    def measure(self, point: np.ndarray, n_samples: int = 5) -> float:
        self.measurement_count += 1
        
        print(f"\n{'='*60}")
        print(f"Measurement #{self.measurement_count}")
        print(f"{'='*60}")
        print(f"Parameters: {point}")
        print(f"Number of samples: {n_samples}")
        print()
        
        measurements = []
        for i in range(n_samples):
            while True:
                try:
                    value_str = input(f"  Sample {i+1}/{n_samples}: ")
                    value = float(value_str)
                    measurements.append(value)
                    break
                except ValueError:
                    print("  Invalid input. Please enter a number.")
                except EOFError:
                    print("\n  Using default value: 0.0")
                    measurements.append(0.0)
                    break
        
        mean_value = float(np.mean(measurements))
        
        self.history.append({
            'point': point.copy(),
            'measurements': measurements.copy(),
            'mean': mean_value
        })
        
        print(f"\n  Measurements: {measurements}")
        print(f"  Mean value: {mean_value:.6f}")
        print(f"{'='*60}\n")
        
        return mean_value
    
    def get_history(self):
        return self.history.copy()
    
    def reset(self):
        self.measurement_count = 0
        self.history.clear()
