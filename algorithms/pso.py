import numpy as np
from typing import List, Tuple
from registry import registry


@registry.register(
    name='algorithm.pso',
    type_='algorithm',
    signature='optimize(bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]'
)
class PSOAlgorithm:
    required_source = 'source.interactive'
    
    def __init__(self, source):
        self.source = source
        self.num_particles = 30
        self.max_iterations = 50
        self.w = 0.7298
        self.c1 = 1.49618
        self.c2 = 1.49618
        self.n_samples = 5
    
    def optimize(self, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        n_dims = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        positions = np.random.uniform(lower, upper, size=(self.num_particles, n_dims))
        velocities = np.random.uniform(-1, 1, size=(self.num_particles, n_dims))
        
        personal_best_positions = positions.copy()
        personal_best_values = np.full(self.num_particles, -np.inf)
        
        global_best_position = None
        global_best_value = -np.inf
        
        for i in range(self.num_particles):
            value = self.source.measure(positions[i], n_samples=self.n_samples)
            personal_best_values[i] = value
            
            if value > global_best_value:
                global_best_value = value
                global_best_position = positions[i].copy()
        
        for iteration in range(self.max_iterations):
            for i in range(self.num_particles):
                r1 = np.random.random(n_dims)
                r2 = np.random.random(n_dims)
                
                cognitive = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social = self.c2 * r2 * (global_best_position - positions[i])
                
                velocities[i] = self.w * velocities[i] + cognitive + social
                
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lower, upper)
                
                value = self.source.measure(positions[i], n_samples=self.n_samples)
                
                if value > personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = positions[i].copy()
                    
                    if value > global_best_value:
                        global_best_value = value
                        global_best_position = positions[i].copy()
        
        return global_best_position, global_best_value
