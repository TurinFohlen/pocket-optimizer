import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from registry import registry


@dataclass
class OptimizationConfig:
    param_bounds: List[Tuple[float, float]]
    param_names:  List[str]
    num_samples:    int   = 5
    max_evaluations: int  = 30
    noise_level:    float = 10.0
    maximize:       bool  = True   # True=最大化（默认）  False=最小化


@dataclass
class HistoryEntry:
    point:     np.ndarray
    value:     float        # 始终是原始值（用户/Source 视角）
    algorithm: str
    timestamp: float
    iteration: int = 0


class SourceWrapper:
    """
    拦截 measure()，做两件事：
    1. 记录历史（存原始值）
    2. 按优化方向翻转返回值（算法始终做最小化）
    """
    def __init__(self, source_instance, orchestrator):
        self.source_instance = source_instance
        self.orchestrator    = orchestrator

    def measure(self, point: np.ndarray) -> float:
        raw = self.source_instance.measure(point)   # 原始值

        if self.orchestrator._current_algorithm:
            import time
            self.orchestrator._evaluation_count += 1
            self.orchestrator.history.append(HistoryEntry(
                point     = point.copy(),
                value     = raw,                    # 存原始值
                algorithm = self.orchestrator._current_algorithm,
                timestamp = time.time(),
                iteration = self.orchestrator._evaluation_count
            ))

        # 算法统一做最大化：
        # 最大化 → 原样传（算法找最大原始值）
        # 最小化 → 取负传（算法找最大负值 = 找最小原始值）
        if self.orchestrator.config.maximize:
            return raw
        return -raw

    def __getattr__(self, name):
        return getattr(self.source_instance, name)


class Orchestrator:
    def __init__(self, config: OptimizationConfig,
                 source_name: Optional[str] = None):
        self.config              = config
        self.source_name         = source_name
        self.history: List[HistoryEntry] = []
        self.source_instance     = None
        self._current_algorithm  = None
        self._evaluation_count   = 0

        if source_name:
            source_class = registry.get_component(source_name)
            if source_class:
                self.source_instance = self._instantiate_source(source_class)

    def _instantiate_source(self, source_class):
        """透传 n_samples（若 source 支持），否则静默用默认值"""
        import inspect
        if 'n_samples' in inspect.signature(source_class.__init__).parameters:
            return source_class(n_samples=self.config.num_samples)
        return source_class()

    def set_source(self, source_name: str):
        self.source_name  = source_name
        source_class = registry.get_component(source_name)
        if source_class is None:
            raise ValueError(f"Source '{source_name}' not found in registry")
        self.source_instance = self._instantiate_source(source_class)

    def run(self, algorithm_name: str) -> Tuple[np.ndarray, float]:
        if self.source_instance is None:
            raise ValueError("No source configured. Call set_source() first.")

        algorithm_class = registry.get_component(algorithm_name)
        if algorithm_class is None:
            raise ValueError(f"Algorithm '{algorithm_name}' not found in registry")

        self._current_algorithm = algorithm_name
        self._evaluation_count  = 0

        wrapped_source   = SourceWrapper(self.source_instance, self)
        algorithm_instance = algorithm_class(source=wrapped_source)

        best_point, best_algo_value = algorithm_instance.optimize(
            self.config.param_bounds
        )

        self._current_algorithm = None

        # 还原为原始值：
        # 最大化 → 算法返回值就是原始值
        # 最小化 → 算法返回的是负值，取负还原
        if self.config.maximize:
            best_raw = best_algo_value
        else:
            best_raw = -best_algo_value

        return best_point, best_raw

    def get_history(self) -> List[HistoryEntry]:
        return self.history.copy()

    def get_history_for_algorithm(self, algorithm_name: str) -> List[HistoryEntry]:
        return [h for h in self.history if h.algorithm == algorithm_name]

    def clear_history(self):
        self.history.clear()
        self._evaluation_count = 0
