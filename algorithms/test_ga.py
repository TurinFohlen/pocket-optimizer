import numpy as np
from typing import Tuple, List
from registry import registry


@registry.register(
    name='algorithm.test_ga',
    type_='algorithm',
    signature='optimize(bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]'
)
class TestGeneticAlgorithm:
    required_source = 'source.test_cli'
    
    def __init__(self, source):
        self.source = source
        self.population_size = 10
        self.generations = 5
    
    def optimize(self, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        n_dims = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        population = np.random.uniform(
            lower, upper, size=(self.population_size, n_dims)
        )
        
        best_point = None
        best_value = -np.inf
        
        for gen in range(self.generations):
            fitness = np.array([
                self.source.measure(ind, n_samples=3)
                for ind in population
            ])
            
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > best_value:
                best_value = fitness[best_idx]
                best_point = population[best_idx].copy()
            
            new_population = []
            for _ in range(self.population_size):
                parents_idx = np.random.choice(
                    self.population_size, size=2, replace=False
                )
                child = 0.5 * (population[parents_idx[0]] + population[parents_idx[1]])
                child += np.random.normal(0, 0.1, n_dims)
                child = np.clip(child, lower, upper)
                new_population.append(child)
            
            population = np.array(new_population)
        
        return best_point, best_value
