import numpy as np
from typing import List, Tuple
from registry import registry


@registry.register(
    name='algorithm.genetic',
    type_='algorithm',
    signature='optimize(bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]'
)
class GeneticAlgorithm:
    required_source = 'source.interactive'
    
    def __init__(self, source):
        self.source = source
        self.population_size = 20
        self.generations = 15
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
    
    def optimize(self, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        n_dims = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        population = np.random.uniform(
            lower, upper, size=(self.population_size, n_dims)
        )
        
        best_point = None
        best_value = -np.inf
        
        for generation in range(self.generations):
            fitness = np.zeros(self.population_size)
            
            for i, individual in enumerate(population):
                value = self.source.measure(individual)
                fitness[i] = value
                
                if value > best_value:
                    best_value = value
                    best_point = individual.copy()
            
            fitness_normalized = fitness - fitness.min() + 1e-10
            probabilities = fitness_normalized / fitness_normalized.sum()
            
            new_population = []
            
            for _ in range(self.population_size // 2):
                parent1_idx = np.random.choice(self.population_size, p=probabilities)
                parent2_idx = np.random.choice(self.population_size, p=probabilities)
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                if np.random.random() < self.crossover_rate:
                    crossover_point = np.random.randint(1, n_dims)
                    child1 = np.concatenate([
                        parent1[:crossover_point],
                        parent2[crossover_point:]
                    ])
                    child2 = np.concatenate([
                        parent2[:crossover_point],
                        parent1[crossover_point:]
                    ])
                else:
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                
                for child in [child1, child2]:
                    if np.random.random() < self.mutation_rate:
                        mutation_idx = np.random.randint(n_dims)
                        child[mutation_idx] = np.random.uniform(
                            lower[mutation_idx],
                            upper[mutation_idx]
                        )
                
                child1 = np.clip(child1, lower, upper)
                child2 = np.clip(child2, lower, upper)
                
                new_population.extend([child1, child2])
            
            population = np.array(new_population[:self.population_size])
        
        return best_point, best_value
