import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from registry import registry


@dataclass
class OptimizationConfig:
    param_bounds: List[Tuple[float, float]]
    param_names: List[str]
    num_samples: int = 5
    max_evaluations: int = 30
    noise_level: float = 10.0


@dataclass
class HistoryEntry:
    point: np.ndarray
    value: float
    algorithm: str
    timestamp: float
    iteration: int = 0


class SourceWrapper:
    def __init__(self, source_instance, orchestrator):
        self.source_instance = source_instance
        self.orchestrator = orchestrator
    
    def measure(self, point: np.ndarray, n_samples: int = 5) -> float:
        value = self.source_instance.measure(point, n_samples)
        
        if self.orchestrator._current_algorithm:
            import time
            self.orchestrator._evaluation_count += 1
            self.orchestrator.history.append(HistoryEntry(
                point=point.copy(),
                value=value,
                algorithm=self.orchestrator._current_algorithm,
                timestamp=time.time(),
                iteration=self.orchestrator._evaluation_count
            ))
        
        return value
    
    def __getattr__(self, name):
        return getattr(self.source_instance, name)


class Orchestrator:
    def __init__(self, config: OptimizationConfig, source_name: Optional[str] = None):
        self.config = config
        self.source_name = source_name
        self.history: List[HistoryEntry] = []
        self.source_instance = None
        self._current_algorithm = None
        self._evaluation_count = 0
        
        if source_name:
            source_class = registry.get_component(source_name)
            if source_class:
                self.source_instance = source_class()
    
    def set_source(self, source_name: str):
        self.source_name = source_name
        source_class = registry.get_component(source_name)
        if source_class is None:
            raise ValueError(f"Source '{source_name}' not found in registry")
        self.source_instance = source_class()
    
    def run(self, algorithm_name: str) -> Tuple[np.ndarray, float]:
        if self.source_instance is None:
            raise ValueError("No source configured. Call set_source() first.")
        
        algorithm_class = registry.get_component(algorithm_name)
        if algorithm_class is None:
            raise ValueError(f"Algorithm '{algorithm_name}' not found in registry")
        
        spec = registry.get_spec(algorithm_name)
        if spec and spec.dependencies:
            missing = [dep for dep in spec.dependencies if dep not in registry.components]
            if missing:
                raise ValueError(f"Missing dependencies: {missing}")
        
        self._current_algorithm = algorithm_name
        self._evaluation_count = 0
        
        wrapped_source = SourceWrapper(self.source_instance, self)
        algorithm_instance = algorithm_class(source=wrapped_source)
        
        best_point, best_value = algorithm_instance.optimize(self.config.param_bounds)
        
        self._current_algorithm = None
        
        return best_point, best_value
    
    def get_history(self) -> List[HistoryEntry]:
        return self.history.copy()
    
    def get_history_for_algorithm(self, algorithm_name: str) -> List[HistoryEntry]:
        return [h for h in self.history if h.algorithm == algorithm_name]
    
    def clear_history(self):
        self.history.clear()
        self._evaluation_count = 0
