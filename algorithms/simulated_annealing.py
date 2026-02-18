import numpy as np
from typing import List, Tuple
from registry import registry


@registry.register(
    name='algorithm.simulated_annealing',
    type_='algorithm',
    signature='optimize(bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]'
)
class SimulatedAnnealingAlgorithm:
    required_source = 'source.interactive'
    
    def __init__(self, source):
        self.source = source
        self.max_iterations = 1000
        self.initial_temperature = 100.0
        self.cooling_rate = 0.95
        self.step_size_initial = 0.5
    
    def optimize(self, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        n_dims = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        span = upper - lower
        
        current_point = np.random.uniform(lower, upper, size=n_dims)
        current_value = self.source.measure(current_point)
        
        best_point = current_point.copy()
        best_value = current_value
        
        temperature = self.initial_temperature
        step_size = self.step_size_initial
        
        for iteration in range(self.max_iterations):
            neighbor = current_point + np.random.normal(0, step_size, n_dims) * span
            neighbor = np.clip(neighbor, lower, upper)
            
            neighbor_value = self.source.measure(neighbor)
            
            delta = neighbor_value - current_value
            
            if delta > 0 or np.random.random() < np.exp(delta / temperature):
                current_point = neighbor
                current_value = neighbor_value
                
                if current_value > best_value:
                    best_value = current_value
                    best_point = current_point.copy()
            
            temperature *= self.cooling_rate
            step_size = self.step_size_initial * (temperature / self.initial_temperature)
            
            if temperature < 1e-8:
                break
        
        return best_point, best_value
