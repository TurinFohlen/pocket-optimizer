from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
import matplotlib.pyplot as plt
import scipy.stats as stats

import pandas
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import warnings
import json
import math
from typing import List, Tuple, Dict, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, asdict
import time
from datetime import datetime
warnings.filterwarnings('ignore')

@dataclass
class OptimizationConfig:
    
    param_bounds: List[Tuple[float, float]]
    param_names: List[str]
    num_samples: int = 5
    precision: int = 2
    gaussian_sigma: float = 0.3
    noise_level: float = 10.0
    max_evaluations: int = 30
    use_advanced_algorithms: bool = True
    exploration_weight: float = 0.3
    exploitation_weight: float = 0.7
    allow_out_of_bounds: bool = True

@dataclass
class EvaluationPoint:
    
    point: np.ndarray
    value: float
    confidence_interval: Tuple[float, float]
    raw_measurements: List[float]
    timestamp: float
    algorithm_used: str
    sample_count: int = 0
    
    def __post_init__(self):
        
        if self.sample_count == 0:
            self.sample_count = len(self.raw_measurements)

@dataclass
class OptimizationResult:
    
    best_point: np.ndarray
    best_value: float
    history: List[EvaluationPoint]
    config: OptimizationConfig
    convergence_rate: Optional[float] = None
    total_time: float = 0.0
    iterations: int = 0

class OptimizationAlgorithm:

    def optimize(self, optimizer: 'EnhancedUniversalOptimizer') -> Tuple[np.ndarray, float]:
        
        raise NotImplementedError

    def get_name(self) -> str:
        
        raise NotImplementedError

class ReflectivePowellAlgorithm(OptimizationAlgorithm):

    def __init__(self, max_iterations: int = 15, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def get_name(self):
        return "Reflective Powell (Mirror)"
    
    def optimize(self, optimizer):
        print(f" 反射Powell算法: 利用镜像映射处理边界")
        
        bounds = optimizer.config.param_bounds
        n_dims = len(bounds)
        
        if optimizer.current_best is not None:
            x0 = optimizer.current_best.copy()
            print(f"   起点: 当前最佳点")
        else:
            x0 = np.array([(b[0] + b[1]) / 2.0 for b in bounds])
            print(f"   起点: 参数空间中心")
        
        def reflect_to_bounds(y):
            
            x = np.zeros_like(y)
            for i in range(n_dims):
                l, h = bounds[i]
                span = h - l
                
                if span <= 0:
                    x[i] = l
                    continue
                
                t = ((y[i] - l) / span) % 2.0
                
                if t < 1.0:
                    x[i] = l + t * span
                else:
                    x[i] = l + (2.0 - t) * span
            return x
        
        def objective_for_scipy(y):
            
            x = reflect_to_bounds(y)
            
            for eval_point in optimizer.history:
                if np.linalg.norm(eval_point.point - x) < self.tolerance:
                    print(f"    反射点命中历史记录,直接使用")
                    return -eval_point.value
            
            eval_point = optimizer.measure(x, "reflective_powell")
            return -eval_point.value
        
        try:
            from scipy.optimize import minimize
            
            print(f"   调用SciPy Powell,最大迭代{self.max_iterations}次")
            result = minimize(
                fun=objective_for_scipy,
                x0=x0,
                method='Powell',
                options={
                    'maxiter': self.max_iterations,
                    'xtol': self.tolerance,
                    'ftol': self.tolerance * 10,
                    'disp': False
                }
            )
            
            best_y = result.x
            best_x = reflect_to_bounds(best_y)
            
            final_point = optimizer.measure(best_x, "reflective_final")
            
            print(f"   反射Powell优化完成。状态: {result.message}")
            print(f"   最终函数调用次数: {result.nfev}")
            
            return best_x, final_point.value
            
        except ImportError:
            print("     SciPy未安装,回退到简单搜索")
            best_point = x0.copy()
            best_value = optimizer.current_best_value if optimizer.current_best is not None else -np.inf
            
            for dim in range(n_dims):
                if not optimizer.should_continue():
                    break
                
                test_point = best_point.copy()
                bounds_dim = bounds[dim]
                step_size = (bounds_dim[1] - bounds_dim[0]) * 0.1
                test_val = best_point[dim] + step_size
                
                if test_val <= bounds_dim[1]:
                    test_point[dim] = test_val
                    eval_point = optimizer.measure(test_point, "powell_fallback")
                    if eval_point.value > best_value:
                        best_value = eval_point.value
                        best_point = test_point.copy()
            
            return best_point, best_value

class BayesianOptimizationAlgorithm(OptimizationAlgorithm):

    def __init__(self, acquisition_function: str = 'ei'):
        self.acquisition_function = acquisition_function

    def get_name(self):
        return f"Bayesian Optimization ({self.acquisition_function.upper()})"

    def optimize(self, optimizer):
        return optimizer._bayesian_optimization_implementation(self.acquisition_function)

class GeneticAlgorithm(OptimizationAlgorithm):

    def __init__(self, population_size: int = 20, generations: int = 15):
        self.population_size = population_size
        self.generations = generations

    def get_name(self):
        return "Genetic Algorithm"

    def optimize(self, optimizer):
        return optimizer._genetic_algorithm_implementation(
            self.population_size, self.generations
        )

class ParticleSwarmAlgorithm(OptimizationAlgorithm):

    def __init__(self, num_particles: int = 10, max_iterations: int = 20):
        self.num_particles = num_particles
        self.max_iterations = max_iterations

    def get_name(self):
        return "Particle Swarm Optimization"

    def optimize(self, optimizer):
        return optimizer._particle_swarm_implementation(
            self.num_particles, self.max_iterations
        )

class SimulatedAnnealingAlgorithm(OptimizationAlgorithm):

    def __init__(self, initial_temp: float = 100.0, cooling_rate: float = 0.95):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def get_name(self):
        return "Simulated Annealing"

    def optimize(self, optimizer):
        return optimizer._simulated_annealing_implementation(
            self.initial_temp, self.cooling_rate
        )

class EnhancedUniversalOptimizer:

    def _init_realtime_plot(self):
        try:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            self.ax.set_title('Optimization Convergence (v9-beta)')
            self.ax.set_xlabel('Evaluations')
            self.ax.set_ylabel('Objective Value')
            self.ax.grid(True, alpha=0.3)
        except:
            pass

    def _update_plot(self):
        if not self.history or not hasattr(self, 'ax'): return
        try:
            self.ax.clear()
            vals = [p.value for p in self.history]
            bests = np.maximum.accumulate(vals)
            self.ax.plot(vals, 'bo-', alpha=0.3, label='Measurement')
            self.ax.plot(bests, 'r-', linewidth=2, label='Best So Far')
            self.ax.legend()
            plt.pause(0.01)
        except: pass

    class _OutlierDetector:
        \"\"\"内部离群检测与邻近搜索模块（封装在Optimizer内部）
        提供统一接口 filter(values, current_point) 被 _robust_statistical_filter 调用。
        \"\"\"

        def __init__(self, optimizer: 'EnhancedUniversalOptimizer'):
            self.optimizer = optimizer

        def normalized_chebyshev_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
            \"\"\"归一化切比雪夫距离（最大维度差异）\"\"\"
            normalized_diffs = []
            for bounds, p1, p2 in zip(self.optimizer.config.param_bounds, point1, point2):
                min_val, max_val = bounds
                range_val = max_val - min_val
                if range_val > 0:
                    normalized_diffs.append(abs(p1 - p2) / range_val)
            return max(normalized_diffs) if normalized_diffs else 0.0

        def get_nearby_points(self, current_point: np.ndarray, radius: float) -> List[EvaluationPoint]:
            \"\"\"切比雪夫半径内的历史点\"\"\"
            nearby = []
            for eval_point in self.optimizer.history:
                dist = self.normalized_chebyshev_distance(current_point, eval_point.point)
                if dist <= radius:
                    nearby.append(eval_point)
            return nearby

        def _average_historical_chebyshev_distance(self) -> float:
            \"\"\"历史点之间的平均切比雪夫距离（采样法）\"\"\"
            history = self.optimizer.history
            if len(history) < 2:
                return 1.0
            distances = []
            points = [ep.point for ep in history]
            n_points = len(points)
            max_pairs = min(30, n_points * (n_points - 1) // 2)
            sampled_pairs = 0
            while sampled_pairs < max_pairs:
                i, j = np.random.choice(n_points, 2, replace=False)
                distances.append(self.normalized_chebyshev_distance(points[i], points[j]))
                sampled_pairs += 1
            return float(np.mean(distances)) if distances else 1.0

        def adaptive_radius(self, current_point: np.ndarray = None, sample_count: int = None) -> float:
            \"\"\"自适应切比雪夫半径（根据历史密度与样本量调整）\"\"\"
            if len(self.optimizer.history) < 3:
                return 0.15
            avg_distance = self._average_historical_chebyshev_distance()
            if avg_distance < 0.05:
                base_radius = 0.05
            elif avg_distance < 0.1:
                base_radius = 0.1
            elif avg_distance < 0.2:
                base_radius = 0.15
            else:
                base_radius = 0.2

            if sample_count is not None and sample_count <= 4:
                boost_map = {1: 3.0, 2: 2.5, 3: 2.0, 4: 1.5}
                base_radius *= boost_map.get(sample_count, 1.0)
                base_radius = min(base_radius, 0.35)

            if current_point is not None:
                test_points = self.get_nearby_points(current_point, base_radius)
                if sample_count is not None and sample_count <= 4:
                    min_points_needed = min(5, len(self.optimizer.history))
                else:
                    min_points_needed = min(3, len(self.optimizer.history))
                expanded = base_radius
                while len(test_points) < min_points_needed and expanded <= 0.4:
                    expanded += 0.05
                    test_points = self.get_nearby_points(current_point, expanded)
                if len(test_points) >= min_points_needed:
                    return expanded
            return base_radius

        def detect_lof_augmented(self, current_samples: List[float], current_point: np.ndarray) -> List[int]:
            \"\"\"增强LOF检测（当样本量小于等于4时可借用邻近点）\"\"\"
            if len(current_samples) < 3:
                return []
            radius = self.adaptive_radius(current_point=current_point, sample_count=len(current_samples))
            nearby = self.get_nearby_points(current_point, radius)
            borrowed = []
            for p in nearby:
                if not np.allclose(p.point, current_point):
                    borrowed.extend(p.raw_measurements)
            combined = current_samples + borrowed if borrowed else list(current_samples)
            if len(combined) < 2:
                return []
            X = np.array(combined).reshape(-1, 1)
            n_neighbors = min(len(X) - 1, 20)
            if n_neighbors < 1:
                return []
            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            is_inlier = lof.fit_predict(X)
            outlier_flags = is_inlier[:len(current_samples)] == -1
            return np.where(outlier_flags)[0].tolist()

        def filter(self, values: np.ndarray, current_point: np.ndarray = None) -> np.ndarray:
            \"\"\"统一的对外过滤接口：对小样本使用增强检测，其它情况不做修改（由上层策略决定）\"\"\"
            sample_size = len(values)
            if sample_size <= 4 and current_point is not None and len(self.optimizer.history) > 0:
                outlier_indices = self.detect_lof_augmented(values.tolist(), current_point)
                if outlier_indices:
                    mask = np.ones(sample_size, dtype=bool)
                    mask[outlier_indices] = False
                    filtered = values[mask]
                    if len(filtered) == 0:
                        median = np.median(values)
                        closest_idx = int(np.argmin(np.abs(values - median)))
                        filtered = values[closest_idx:closest_idx+1]
                    return filtered
                else:
                    return values
            return values
    def detect_outliers_lof_augmented(self, current_samples, current_point):
        
        if len(current_samples) < 3:
            return []
        nearby = self.get_nearby_points_sklearn(current_point, n_neighbors=10)
        borrowed = []
        for p in nearby:
            if not np.allclose(p.point, current_point):
                borrowed.extend(p.raw_measurements)
        if borrowed:
            combined = current_samples + borrowed
        else:
            combined = current_samples[:]
        if len(combined) < 2:
            return []
        X = np.array(combined).reshape(-1, 1)
        n_neighbors = min(len(X) - 1, 20)
        if n_neighbors < 1:
            return []
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        is_inlier = lof.fit_predict(X)
        outlier_flags = is_inlier[:len(current_samples)] == -1
        return np.where(outlier_flags)[0].tolist()

    def __init__(self, config: OptimizationConfig):
        
        self.config = config
        self.n_dims = len(config.param_bounds)

        self.all_history: List[EvaluationPoint] = []
        self.history: List[EvaluationPoint] = []
        
        self.global_best: Optional[np.ndarray] = None
        self.global_best_value: float = -np.inf
        self.current_best: Optional[np.ndarray] = None
        self.current_best_value: float = -np.inf
        
        self.start_time: float = time.time()
        self.iterations: int = 0

        self.algorithms: Dict[str, OptimizationAlgorithm] = {
            "powell": ReflectivePowellAlgorithm(),
            "bayesian_ei": BayesianOptimizationAlgorithm('ei'),
            "bayesian_ucb": BayesianOptimizationAlgorithm('ucb'),
            "genetic": GeneticAlgorithm(),
            "particle_swarm": ParticleSwarmAlgorithm(),
            "simulated_annealing": SimulatedAnnealingAlgorithm(),
        }

        self.algorithm_usage: Dict[str, int] = {}

        print(f" 优化器初始化完成")
        print(f"   参数维度: {self.n_dims}")
        print(f"   参数范围: {config.param_bounds}")
        print(f"   可用算法: {list(self.algorithms.keys())}")
        if config.allow_out_of_bounds:
            print(f"     允许超界数据: 是 (所有历史数据将被保留用于训练)")
        self._init_realtime_plot()   
    
        self._outlier_detector = self._OutlierDetector(self)
    
    def is_in_current_bounds(self, point: np.ndarray) -> bool:
        
        for i, (val, bounds) in enumerate(zip(point, self.config.param_bounds)):
            if val < bounds[0] or val > bounds[1]:
                return False
        return True

    def normalized_chebyshev_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        
        normalized_diffs = []
        for i, (bounds, p1, p2) in enumerate(zip(self.config.param_bounds, point1, point2)):
            min_val, max_val = bounds
            range_val = max_val - min_val
            
            if range_val > 0:
                norm_diff = abs(p1 - p2) / range_val
                normalized_diffs.append(norm_diff)
        
        return max(normalized_diffs) if normalized_diffs else 0.0
    
    def get_nearby_points_chebyshev(self, current_point: np.ndarray, radius: float) -> List[EvaluationPoint]:
        
        nearby = []
        for eval_point in self.history:
            dist = self.normalized_chebyshev_distance(current_point, eval_point.point)
            if dist <= radius:
                nearby.append(eval_point)
        return nearby
    
    def _average_historical_chebyshev_distance(self) -> float:
        
        if len(self.history) < 2:
            return 1.0
        
        distances = []
        points = [eval_point.point for eval_point in self.history]
        n_points = len(points)
        
        max_pairs = min(30, n_points * (n_points - 1) // 2)
        sampled_pairs = 0
        
        while sampled_pairs < max_pairs:
            i, j = np.random.choice(n_points, 2, replace=False)
            dist = self.normalized_chebyshev_distance(points[i], points[j])
            distances.append(dist)
            sampled_pairs += 1
        
        return np.mean(distances) if distances else 1.0
    
    def adaptive_chebyshev_radius(self, current_point: np.ndarray = None, sample_count: int = None) -> float:
        
        if len(self.history) < 3:
            return 0.15
        
        avg_distance = self._average_historical_chebyshev_distance()
        
        if avg_distance < 0.05:
            base_radius = 0.05
        elif avg_distance < 0.1:
            base_radius = 0.1
        elif avg_distance < 0.2:
            base_radius = 0.15
        else:
            base_radius = 0.2
        
        if sample_count is not None and sample_count <= 4:
            if sample_count == 1:
                boost_factor = 3.0
            elif sample_count == 2:
                boost_factor = 2.5
            elif sample_count == 3:
                boost_factor = 2.0
            else:
                boost_factor = 1.5
            
            boosted_radius = min(base_radius * boost_factor, 0.35)
            print(f"     样本数={sample_count}，半径扩大: {base_radius:.0%} → {boosted_radius:.0%}")
            base_radius = boosted_radius
        
        if current_point is not None:
            test_points = self.get_nearby_points_chebyshev(current_point, base_radius)
            
            if sample_count is not None and sample_count <= 4:
                min_points_needed = min(5, len(self.history))
            else:
                min_points_needed = min(3, len(self.history))
            
            if len(test_points) < min_points_needed:
                expanded_radius = base_radius
                while expanded_radius <= 0.4 and len(test_points) < min_points_needed:
                    expanded_radius += 0.05
                    test_points = self.get_nearby_points_chebyshev(current_point, expanded_radius)
                
                if len(test_points) >= min_points_needed:
                    print(f"    自适应半径: {base_radius:.0%} → {expanded_radius:.0%} (找到 {len(test_points)} 个邻近点)")
                    return expanded_radius
        
        return base_radius

    def measure(self, point: np.ndarray, algorithm_name: str = "manual") -> EvaluationPoint:
        
        print(f"\n 测量点: {self._format_point(point)}")
        print(f"   使用算法: {algorithm_name}")

        values = []
        for i in range(self.config.num_samples):
            while True:
                try:
                    val_input = input(f"   第 {i+1}/{self.config.num_samples} 次测量值: ")
                    val = float(val_input)
                    values.append(val)
                    break
                except ValueError:
                    print(f"   错误: '{val_input}' 不是有效数字，请重新输入")

        values_array = np.array(values)
        sample_count = len(values)

        filtered_values = self._robust_statistical_filter(values_array, current_point=point)

        n_filtered = len(filtered_values)
        n_original = len(values_array)
        if n_filtered < n_original:
            print(f"   警告: 过滤掉 {n_original - n_filtered} 个异常值")

        mean_val = self._robust_mean_estimate(filtered_values)

        if n_filtered >= 2:
            ci_lower, ci_upper = self._calculate_confidence_interval(filtered_values)
        else:
            ci_lower = ci_upper = mean_val
            print(f"   警告: 有效样本不足 ({n_filtered})，置信区间不可靠")

        mean_val = round(mean_val, self.config.precision)
        ci_lower = round(ci_lower, self.config.precision)
        ci_upper = round(ci_upper, self.config.precision)

        eval_point = EvaluationPoint(
            point=point.copy(),
            value=mean_val,
            confidence_interval=(ci_lower, ci_upper),
            raw_measurements=values,
            timestamp=time.time(),
            algorithm_used=algorithm_name,
            sample_count=sample_count
        )

        self.all_history.append(eval_point)
        
        in_bounds = self.is_in_current_bounds(point)
        
        if in_bounds:
            self.history.append(eval_point)
            
            if mean_val > self.current_best_value:
                self.current_best_value = mean_val
                self.current_best = point.copy()
                print(f"    发现新的最佳值(当前范围): {mean_val:.{self.config.precision}f}")
        else:
            print(f"     注意: 此点超出当前参数范围,但数据已保留用于模型训练")
        
        if mean_val > self.global_best_value:
            self.global_best_value = mean_val
            self.global_best = point.copy()
            if not in_bounds:
                print(f"   ⭐ 发现新的全局最佳值: {mean_val:.{self.config.precision}f} (超出当前范围)")

        print(f"   有效样本数: {n_filtered}/{n_original}")
        print(f"   稳健均值: {mean_val:.{self.config.precision}f}")
        if n_filtered >= 2:
            ci_width = ci_upper - ci_lower
            relative_width = (ci_width / abs(mean_val + 1e-10)) * 100
            print(f"   95% 置信区间: [{ci_lower:.{self.config.precision}f}, {ci_upper:.{self.config.precision}f}]")
            print(f"   区间宽度: {ci_width:.{self.config.precision}f} ({relative_width:.1f}%)")

        self._update_plot()
        return eval_point

    def _robust_statistical_filter(self, values: np.ndarray, current_point: np.ndarray = None) -> np.ndarray:
        
        sample_size = len(values)
        
        if sample_size <= 4:
            if current_point is not None and len(self.history) > 0:
                outlier_indices = self.detect_outliers_lof_augmented(values.tolist(), current_point)

                if outlier_indices:
                    print(f"    检测到 {len(outlier_indices)} 个离群点: {outlier_indices}")
                    mask = np.ones(len(values), dtype=bool)
                    mask[outlier_indices] = False
                    filtered = values[mask]
                    
                    if len(filtered) == 0:
                        print(f"     所有样本都被标记为离群，保留离中位数最近的样本")                 
                        median = np.median(values)                 
                        closest_idx = np.argmin(np.abs(values - median))
                        filtered = values[closest_idx:closest_idx+1]
                    
                    print(f"   过滤结果: {len(filtered)}/{sample_size} 个样本被保留")
                    return filtered
                else:
                    print(f"   样本量 {sample_size} ≤ 4，未检测到离群点")
                    return values
            else:
                print(f"   样本量 {sample_size} ≤ 4，跳过异常值检测（无历史数据）")
                return values
sample_size = len(values)

if sample_size <= 4:
    if current_point is not None and len(self.history) > 0:
        filtered = self._outlier_detector.filter(values, current_point)
        n_filtered = len(filtered)
        if n_filtered < sample_size:
            print(f"    检测并过滤了 {sample_size - n_filtered} 个离群样本 (通过 _OutlierDetector)")
        print(f"   过滤结果: {n_filtered}/{sample_size} 个样本被保留")
        return filtered
    else:
        print(f"   样本量 {sample_size} ≤ 4，跳过异常值检测（无历史数据）")
        return values

        elif sample_size <= 16:
            methods = ['iqr', 'percentile']
            print(f"   样本量 {sample_size}，使用稳健策略: {methods}")
        
        elif sample_size <= 32:
            methods = ['iqr', 'modified_zscore', 'percentile']
            print(f"   样本量 {sample_size}，使用中等策略: {methods}")
        
        elif sample_size <= 64:
            methods = ['iqr', 'modified_zscore', 'tukey', 'percentile']
            print(f"   样本量 {sample_size}，使用强策略: {methods}")
        
        else:
            methods = ['iqr', 'zscore', 'modified_zscore', 'tukey', 'percentile']
            print(f"   样本量 {sample_size}，使用严格策略: {methods}")
        
        mask = np.ones_like(values, dtype=bool)
        
        for method in methods:
            if method == 'iqr':
                Q1 = np.percentile(values, 25)
                Q3 = np.percentile(values, 75)
                IQR = max(Q3 - Q1, 1e-10)

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                mask &= (values >= lower_bound) & (values <= upper_bound)

            elif method == 'modified_zscore':
                median = np.median(values)
                mad = np.median(np.abs(values - median))

                if mad > 0:
                    modified_z = 0.6745 * (values - median) / mad
                    mask &= (np.abs(modified_z) < 3.5)

            elif method == 'zscore':
                mean_val = np.mean(values)
                std_val = np.std(values)
                if std_val > 0:
                    z_scores = np.abs((values - mean_val) / std_val)
                    mask &= (z_scores < 3)

            elif method == 'percentile':
                lower_bound = np.percentile(values, 5)
                upper_bound = np.percentile(values, 95)
                mask &= (values >= lower_bound) & (values <= upper_bound)

            elif method == 'tukey':
                Q1 = np.percentile(values, 25)
                Q3 = np.percentile(values, 75)
                IQR = max(Q3 - Q1, 1e-10)

                lower_bound = Q1 - 3.0 * IQR
                upper_bound = Q3 + 3.0 * IQR
                mask &= (values >= lower_bound) & (values <= upper_bound)
        
        filtered = values[mask]
        
        min_samples_needed = min(2, sample_size)
        
        if len(filtered) < min_samples_needed:
            print(f"   警告: 过滤后样本过少 ({len(filtered)}/{sample_size})，恢复部分样本")
            
            if len(filtered) == 0:
                if sample_size >= 2:
                    median = np.median(values)
                    distances = np.abs(values - median)
                    idx = np.argsort(distances)[:min(2, sample_size)]
                    filtered = values[idx]
                    print(f"   恢复离中位数最近的 {len(filtered)} 个样本")
                else:
                    filtered = values.copy()
            else:
                remaining_needed = min_samples_needed - len(filtered)
                median = np.median(values)
                distances = np.abs(values - median)
                
                sorted_idx = np.argsort(distances)
                for idx in sorted_idx:
                    if idx not in np.where(mask)[0] and remaining_needed > 0:
                        filtered = np.append(filtered, values[idx])
                        remaining_needed -= 1
        
        print(f"   过滤结果: {len(filtered)}/{sample_size} 个样本被保留")
        return filtered

    def _robust_mean_estimate(self, values: np.ndarray) -> float:
        
        n = len(values)
        if n == 0:
            return 0.0
        elif n == 1:
            return float(values[0])
        elif n == 2:
            return np.mean(values)
        else:
            trim_proportion = min(0.1, 0.5/n)
            from scipy import stats

            trimmed_mean = stats.trim_mean(values, trim_proportion)
            return float(trimmed_mean)

    def _calculate_confidence_interval(self, values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        
        n = len(values)
        if n < 2:
            mean_val = np.mean(values) if n == 1 else 0.0
            return mean_val, mean_val

        mean_val = np.mean(values)

        try:
            if n < 30:
                std_val = np.std(values, ddof=1)
                std_err = std_val / np.sqrt(n)

                from scipy import stats
                t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
                margin = t_critical * std_err

                ci_lower = mean_val - margin
                ci_upper = mean_val + margin

            else:
                std_val = np.std(values, ddof=1)
                std_err = std_val / np.sqrt(n)

                from scipy import stats
                z_critical = stats.norm.ppf((1 + confidence) / 2)
                margin = z_critical * std_err

                ci_lower = mean_val - margin
                ci_upper = mean_val + margin

            if n >= 10 and n < 100:
                bootstrap_cis = self._bootstrap_confidence_interval(values, confidence)
                bootstrap_width = bootstrap_cis[1] - bootstrap_cis[0]
                t_width = ci_upper - ci_lower

                if abs(bootstrap_width - t_width) / (t_width + 1e-10) > 0.5:
                    print(f"   注意: 使用Bootstrap置信区间（分布可能非正态）")
                    return bootstrap_cis

            return ci_lower, ci_upper

        except Exception as e:
            print(f"   警告: 置信区间计算失败: {e}, 使用百分位数区间")
            lower_percentile = (1 - confidence) / 2 * 100
            upper_percentile = (1 + confidence) / 2 * 100
            return np.percentile(values, lower_percentile), np.percentile(values, upper_percentile)

    def _bootstrap_confidence_interval(self, values: np.ndarray, confidence: float = 0.95,
                                     n_bootstrap: int = 1000) -> Tuple[float, float]:
        
        n = len(values)
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100

        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)

        return ci_lower, ci_upper

    def _format_point(self, point: np.ndarray) -> str:
        
        formatted = []
        for val, name in zip(point, self.config.param_names):
            formatted.append(f"{name}={val:.{self.config.precision}f}")
            
        return ", ".join(formatted)

    def create_initial_samples(self, n_samples: int = 5) -> List[np.ndarray]:
        
        n_dims = len(self.config.param_bounds)
        samples = []

        unit_samples = np.zeros((n_samples, n_dims))

        for d in range(n_dims):
            intervals = np.linspace(0, 1, n_samples + 1)
            points = np.random.uniform(intervals[:-1], intervals[1:])
            
            np.random.shuffle(points)
            unit_samples[:, d] = points

        for i in range(n_samples):
            real_point = []
            for d in range(n_dims):
                min_val, max_val = self.config.param_bounds[d]
                val = min_val + unit_samples[i, d] * (max_val - min_val)
                real_point.append(val)
            samples.append(np.array(real_point))

        return samples

    def should_continue(self) -> bool:
        
        if len(self.history) >= self.config.max_evaluations:
            print(f" 达到最大评估次数限制: {self.config.max_evaluations}")
            return False

        if len(self.history) >= 16:
            recent_values = [p.value for p in self.history[-5:]]
            improvement = max(recent_values) - min(recent_values)

            convergence_threshold = self.config.noise_level * 0.3
            
            if improvement < convergence_threshold:
                print(f" 检测到收敛: 近期改进 {improvement:.3f} < 阈值 {convergence_threshold:.3f}")
                return False

        return True

    def select_algorithm(self) -> OptimizationAlgorithm:

        if hasattr(self, 'force_algorithm') and self.force_algorithm is not None:
            algo_map = {
                'latin_hypercube': 'genetic',
                'genetic': 'genetic',
                'powell': 'powell',
                'particle_swarm': 'particle_swarm',
                'simulated_annealing': 'simulated_annealing',
                'bayesian_ei': 'bayesian_ei',
                'bayesian_ucb': 'bayesian_ucb'
            }
            
            if self.force_algorithm in algo_map:
                algo_key = algo_map[self.force_algorithm]
                algo_name = self.algorithms[algo_key].get_name()
                print(f" 用户指定算法: {algo_name}")
                return self.algorithms[algo_key]

        if len(self.history) < 12:
            print(" 初始探索阶段，使用遗传算法")
            return self.algorithms["genetic"]

        elif len(self.history) < 24:
            print(" 早中期阶段，使用粒子群优化")
            return self.algorithms["particle_swarm"]

        elif len(self.history) < 32:
            if np.random.random() < 0.5:
                print(" 中期阶段，使用模拟退火算法")
                return self.algorithms["simulated_annealing"]
            else:
                print(" 中期阶段，使用Powell算法")
                return self.algorithms["powell"]

        else:
            print(" 精细优化阶段，使用贝叶斯优化")
            if np.random.random() < 0.5:
                return self.algorithms["bayesian_ei"]
            else:
                return self.algorithms["bayesian_ucb"]

    def _bayesian_optimization_implementation(self, acquisition_function: str):
        
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
            from sklearn.preprocessing import StandardScaler

            if len(self.history) < 3:
                print("历史数据不足，使用拉丁超立方采样")
                point = self.create_initial_samples(1)[0]
                eval_point = self.measure(point, "bayesian_random")
                return point, eval_point.value

            X = np.array([p.point for p in self.history])
            y = np.array([p.value for p in self.history])

            noise_levels = []
            valid_noise_data = 0

            for p in self.history:
                if hasattr(p, 'raw_measurements') and len(p.raw_measurements) >= 2:
                    noise_var = np.var(p.raw_measurements) if len(p.raw_measurements) > 1 else 0.0
                    noise_levels.append(noise_var)
                    valid_noise_data += 1

            if self.n_dims <= 3:
                base_kernel = RBF(
                    length_scale=np.ones(self.n_dims),
                    length_scale_bounds=(1e-2, 1e2)
                )
            else:
                base_kernel = Matern(
                    nu=2.5,
                    length_scale=np.ones(self.n_dims),
                    length_scale_bounds=(1e-2, 1e2)
                )

            constant_kernel = ConstantKernel(
                constant_value=1.0,
                constant_value_bounds=(1e-3, 1e3)
            )

            if valid_noise_data >= 3 and np.mean(noise_levels) > 1e-10:
                avg_noise = np.mean(noise_levels)
                print(f"使用估计的噪声水平: {avg_noise:.4f}")
                noise_kernel = WhiteKernel(
                    noise_level=avg_noise,
                    noise_level_bounds=(avg_noise * 0.1, avg_noise * 10)
                )
            else:
                noise_kernel = WhiteKernel(
                    noise_level=1.0,
                    noise_level_bounds=(1e-10, 1e2)
                )

            kernel = constant_kernel * base_kernel + noise_kernel

            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                alpha=1e-6,
                normalize_y=True,
                random_state=42
            )

            X_scaler = StandardScaler()
            X_scaled = X_scaler.fit_transform(X)

            print(f"训练高斯过程... 使用 {len(X)} 个数据点")
            try:
                gp.fit(X_scaled, y)
                print(f"核函数参数: {gp.kernel_}")
                print(f"对数边际似然: {gp.log_marginal_likelihood():.2f}")
            except Exception as e:
                print(f"高斯过程训练失败: {e}")
                simple_kernel = RBF() + WhiteKernel()
                gp = GaussianProcessRegressor(kernel=simple_kernel)
                gp.fit(X_scaled, y)

            n_candidates = min(500, 100 * self.n_dims)

            print(f"生成 {n_candidates} 个候选点...")
            candidates = np.array(self.create_initial_samples(n_candidates))

            candidates_scaled = X_scaler.transform(candidates)

            print("预测候选点...")
            mu, sigma = gp.predict(candidates_scaled, return_std=True)

            sigma = np.maximum(sigma, 1e-9)

            if acquisition_function == 'ei':
                 best_y = np.max(y)
                 z = (mu - best_y) / sigma
                 from scipy.stats import norm
                 scores = (mu - best_y) * norm.cdf(z) + sigma * norm.pdf(z)
            else:
                 scores = mu + 1.96 * sigma

            valid_indices = np.isfinite(scores)
            if not np.any(valid_indices):
                print("所有采集函数值无效，使用随机选择")
                best_idx = np.random.randint(len(candidates))
            else:
                scores = np.where(valid_indices, scores, -np.inf)
                best_idx = np.argmax(scores)

            candidate_point = candidates[best_idx]

            for i in range(self.n_dims):
                low, high = self.config.param_bounds[i]
                candidate_point[i] = np.clip(candidate_point[i], low, high)

            print(f"选择候选点: {self._format_point(candidate_point)}")
            print(f"预测均值: {mu[best_idx]:.{self.config.precision}f}")
            print(f"预测标准差: {sigma[best_idx]:.{self.config.precision}f}")

            eval_point = self.measure(candidate_point, f"bayesian_{acquisition_function}")
            return candidate_point, eval_point.value

        except Exception as e:
            print(f"贝叶斯优化失败: {e}")
            import traceback
            traceback.print_exc()

            print("尝试回退策略...")
            if self.current_best is not None:
                perturbation_point = self.current_best.copy()
                for i in range(self.n_dims):
                    low, high = self.config.param_bounds[i]
                    range_size = high - low
                    perturbation = np.random.normal(0, range_size * 0.05)
                    perturbation_point[i] = np.clip(
                        perturbation_point[i] + perturbation, low, high
                    )
                eval_point = self.measure(perturbation_point, "bayesian_fallback")
                return perturbation_point, eval_point.value
            else:
                point = self.create_initial_samples(1)[0]
                eval_point = self.measure(point, "random_fallback")
                return point, eval_point.value

    def _genetic_algorithm_implementation(self, population_size: int, generations: int):
        
        print(f" 遗传算法:           种群大小={population_size}, 最大代数={generations}")

        population = np.array(self.create_initial_samples(population_size))
        fitness = np.full(population_size, -np.inf)

        best_point = None
        best_value = -np.inf

        print("   评估初始种群...")
        for i, point in enumerate(population):
            if not self.should_continue():
                return best_point, best_value
            eval_point = self.measure(point, "genetic_init")
            fitness[i] = eval_point.value
            if fitness[i] > best_value:
                best_value = fitness[i]
                best_point = point.copy()

        tournament_size = 3
        crossover_prob = 0.8
        mutation_prob = 0.1
        mutation_strength = 0.05

        for gen in range(1, generations + 1):
            print(f"   第 {gen}/{generations} 代")

            if not self.should_continue():
                break

            new_population = []
            new_fitness = []

            elite_idx = np.argmax(fitness)
            new_population.append(population[elite_idx].copy())
            new_fitness.append(fitness[elite_idx])

            while len(new_population) < population_size:
                def tournament_select():
                    candidates = np.random.choice(len(population), tournament_size, replace=False)
                    winner_idx = candidates[np.argmax(fitness[candidates])]
                    return population[winner_idx].copy()

                p1 = tournament_select()
                p2 = tournament_select()

                if np.random.rand() < crossover_prob:
                    alpha = np.random.uniform(0.3, 0.7)
                    child1 = alpha * p1 + (1 - alpha) * p2
                    child2 = (1 - alpha) * p1 + alpha * p2
                else:
                    child1, child2 = p1.copy(), p2.copy()

                for child in (child1, child2):
                    if np.random.rand() < mutation_prob:
                        for d in range(self.n_dims):
                            low, high = self.config.param_bounds[d]
                            sigma = (high - low) * mutation_strength
                            child[d] += np.random.normal(0, sigma)
                            child[d] = np.clip(child[d], low, high)

                new_population.append(child1)
                new_fitness.append(-np.inf)
                if len(new_population) < population_size:
                    new_population.append(child2)
                    new_fitness.append(-np.inf)

            new_population = new_population[:population_size]
            new_fitness = new_fitness[:population_size]

            for i, point in enumerate(new_population):
                if not self.should_continue():
                    return best_point, best_value
                eval_point = self.measure(point, "genetic")
                new_fitness[i] = eval_point.value
                if new_fitness[i] > best_value:
                    best_value = new_fitness[i]
                    best_point = point.copy()

            population = np.array(new_population)
            fitness = np.array(new_fitness)

        return best_point, best_value

    def _particle_swarm_implementation(self, num_particles: int, max_iterations: int):
        
        print(f" 粒子群优化: 粒子数={num_particles}, 最大迭代={max_iterations}")

        positions = np.array(self.create_initial_samples(num_particles))
        velocities = np.zeros_like(positions)

        pbest_pos = positions.copy()
        pbest_val = np.full(num_particles, -np.inf)

        gbest_pos = None
        gbest_val = -np.inf

        print("   评估初始粒子...")
        for i in range(num_particles):
            if not self.should_continue():
                return gbest_pos, gbest_val
            eval_point = self.measure(positions[i], "pso_init")
            pbest_val[i] = eval_point.value
            if pbest_val[i] > gbest_val:
                gbest_val = pbest_val[i]
                gbest_pos = positions[i].copy()

        w_max = 0.9
        w_min = 0.4
        c1 = 2.0
        c2 = 2.0

        for it in range(1, max_iterations + 1):
            print(f"   迭代 {it}/{max_iterations}")

            if not self.should_continue():
                break

            w = w_max - (w_max - w_min) * (it / max_iterations)

            for i in range(num_particles):
                r1 = np.random.rand(self.n_dims)
                r2 = np.random.rand(self.n_dims)

                velocities[i] = (
                    w * velocities[i] +
                    c1 * r1 * (pbest_pos[i] - positions[i]) +
                    c2 * r2 * (gbest_pos - positions[i])
                )

                positions[i] += velocities[i]

                for d in range(self.n_dims):
                    low, high = self.config.param_bounds[d]
                    if positions[i][d] < low or positions[i][d] > high:
                        positions[i][d] = np.clip(positions[i][d], low, high)
                        velocities[i][d] *= -0.5

                if not self.should_continue():
                    return gbest_pos, gbest_val

                eval_point = self.measure(positions[i], "pso")
                fitness = eval_point.value

                if fitness > pbest_val[i]:
                    pbest_val[i] = fitness
                    pbest_pos[i] = positions[i].copy()

                if fitness > gbest_val:
                    gbest_val = fitness
                    gbest_pos = positions[i].copy()

        return gbest_pos, gbest_val

    def _simulated_annealing_implementation(self, initial_temp: float, cooling_rate: float):
        
        print(f" 模拟退火: 初始温度={initial_temp}, 冷却率={cooling_rate}")

        if self.current_best is None:
            current_point = self.create_initial_samples(1)[0]
            eval_point = self.measure(current_point, "simulated_annealing")
            current_value = eval_point.value
        else:
            current_point = self.current_best.copy()
            current_value = self.current_best_value

        best_point = current_point.copy()
        best_value = current_value

        temp = initial_temp
        iteration = 0
        max_iterations = 5

        while temp > 1.0 and iteration < max_iterations and self.should_continue():
            iteration += 1
            print(f"   迭代 {iteration}, 温度={temp:.2f}")

            neighbor = current_point.copy()
            for dim in range(self.n_dims):
                bounds = self.config.param_bounds[dim]
                range_size = bounds[1] - bounds[0]
                perturbation = np.random.normal(0, range_size * 0.1 * (temp / initial_temp))
                neighbor[dim] = np.clip(current_point[dim] + perturbation, bounds[0], bounds[1])

            eval_point = self.measure(neighbor, "simulated_annealing")
            neighbor_value = eval_point.value

            delta = neighbor_value - current_value
            if delta > 0 or np.random.random() < np.exp(delta / temp):
                current_point = neighbor.copy()
                current_value = neighbor_value
                print(f"    接受新点 (delta={delta:.2f})")

                if neighbor_value > best_value:
                    best_point = neighbor.copy()
                    best_value = neighbor_value
                    print(f"    更新最佳点")

            temp *= cooling_rate

        return best_point, best_value
        
    def run_optimization(self) -> OptimizationResult:
        
        print("\n" + "="*80)
        print(" 开始优化流程")
        print("="*80)

        if len(self.history) == 0:
            print("\n 阶段1: 初始探索")
            initial_points = self.create_initial_samples(3)

            for point in initial_points:
                self.measure(point, "initial_exploration")

        self.iterations = 0

        while self.should_continue():
            self.iterations += 1

            print(f"\n 迭代 {self.iterations}")
            print("-"*40)

            algorithm = self.select_algorithm()
            algorithm_name = algorithm.get_name()

            if algorithm_name in self.algorithm_usage:
                self.algorithm_usage[algorithm_name] += 1
            else:
                self.algorithm_usage[algorithm_name] = 1

            try:
                best_point, best_value = algorithm.optimize(self)
                print(f"   算法 {algorithm_name} 完成")
                print(f"   当前最佳值: {best_value:.{self.config.precision}f}")
            except Exception as e:
                print(f" 算法 {algorithm_name} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        total_time = time.time() - self.start_time

        result = OptimizationResult(
            best_point=self.current_best,
            best_value=self.current_best_value,
            history=self.history,
            config=self.config,
            total_time=total_time,
            iterations=self.iterations
        )

        return result

    def analyze_results(self, result: OptimizationResult):
        
        print("\n" + "="*80)
        print(" 优化结果分析")
        print("="*80)

        print(f"\n 最优参数组合:")
        print(f"   {self._format_point(result.best_point)}")
        print(f"   预期值: {result.best_value:.{self.config.precision}f}")

        if result.best_point is not None and len(result.history) > 0:
            for eval_point in result.history:
                if np.array_equal(eval_point.point, result.best_point):
                    ci_lower, ci_upper = eval_point.confidence_interval
                    print(f"   95% 置信区间: [{ci_lower:.{self.config.precision}f}, {ci_upper:.{self.config.precision}f}]")
                    break

        print(f"\n 统计信息:")
        print(f"   总测试点数: {len(result.history)}")
        print(f"   总迭代次数: {result.iterations}")
        print(f"   总耗时: {result.total_time:.2f} 秒")

        if len(result.history) > 0:
            first_value = result.history[0].value
            improvement = result.best_value - first_value
            print(f"   相对于初始点的改进: {improvement:.{self.config.precision}f}")

        print(f"\n 算法使用统计:")
        for algo_name, count in self.algorithm_usage.items():
            print(f"   {algo_name}: {count} 次")

        if len(result.history) >= 5:
            print(f"\n 参数敏感度分析:")

            for dim in range(self.n_dims):
                param_name = self.config.param_names[dim]
                param_values = [p.point[dim] for p in result.history]
                param_values = np.array(param_values)

                if len(np.unique(param_values)) > 2:
                    objective_values = [p.value for p in result.history]
                    correlation = np.corrcoef(param_values, objective_values)[0, 1]

                    if not np.isnan(correlation):
                        print(f"   {param_name}: 相关系数 = {correlation:.3f}")

    def export_results(self, result: OptimizationResult, filename: str = "optimization_results.json",
                       export_all: bool = False):

        if export_all:
            history_to_export = self.all_history
            export_scope = "all"
        else:
            history_to_export = result.history
            export_scope = "current_range"
        
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "optimizer_version": "2.0.4",
                "export_scope": export_scope,
                "total_evaluations": len(history_to_export),
                "total_time": result.total_time,
                "iterations": result.iterations
            },
            "configuration": asdict(self.config),
            "best_result": {
                "point": result.best_point.tolist() if result.best_point is not None else None,
                "value": float(result.best_value),
                "formatted": self._format_point(result.best_point) if result.best_point is not None else None
            },
            "history": [
                {
                    "point": ep.point.tolist(),
                    "value": ep.value,
                    "confidence_interval": ep.confidence_interval,
                    "sample_count": getattr(ep, 'sample_count', len(ep.raw_measurements)),
                    "algorithm_used": ep.algorithm_used,
                    "timestamp": ep.timestamp
                }
                for ep in history_to_export
            ],
            "algorithm_statistics": self.algorithm_usage,
            "analysis": {
                "parameter_sensitivity": self._calculate_parameter_sensitivity(result)
            }
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n 结果已导出到: {filename}")
        if export_all:
            print(f"   导出范围: 所有历史数据 ({len(history_to_export)}条)")
        else:
            print(f"   导出范围: 当前参数范围内 ({len(history_to_export)}条)")

    def export_results_wl(self, result: OptimizationResult, filename: str = "optimization_results.wl", 
                          export_all: bool = False):

        if export_all:
            history_to_export = self.all_history
            export_note = "All History Data (包括超出当前范围的数据)"
        else:
            history_to_export = result.history
            export_note = "Current Range Data (仅当前参数范围内的数据)"
        
        def escape_string(s):
            
            if s is None:
                return '""'
            escaped = str(s).replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        
        def format_number(num):
            
            if num is None:
                return "None"
            return f"{num:.{self.config.precision}f}"
        
        def format_timestamp(t):
            
            if t is None:
                return "None"
            return f"AbsoluteTime[1970] + {t}"
        
        def format_confidence_interval(ci):
            
            if ci is None:
                return "None"
            try:
                lower, upper = ci
                return f"{{{format_number(lower)}, {format_number(upper)}}}"
            except (TypeError, ValueError):
                return "None"
        
        wl_content = []
        
        wl_content.append("(* Optimization Results *)")
        wl_content.append(f"(* Export Time: {datetime.now().isoformat()} *)")
        wl_content.append(f"(* Optimizer Version: 2.0.4 *)")
        wl_content.append(f"(* Export Scope: {export_note} *)")
        wl_content.append(f"(* Total Evaluations: {len(history_to_export)} *)")
        wl_content.append(f"(* Sample Enhancement: Enabled *)")
        wl_content.append("")
        
        wl_content.append("(* Configuration *)")
        wl_content.append("config = Association[")
        wl_content.append(f'  "paramBounds" -> {self._format_wl_bounds(self.config.param_bounds)},')
        wl_content.append(f'  "paramNames" -> {self._format_wl_list(self.config.param_names)},')
        wl_content.append(f'  "numSamples" -> {self.config.num_samples},')
        wl_content.append(f'  "precision" -> {self.config.precision},')
        wl_content.append(f'  "maxEvaluations" -> {self.config.max_evaluations}')
        wl_content.append("];")
        wl_content.append("")
        
        wl_content.append("(* Best Result *)")
        if result.best_point is not None:
            wl_content.append("bestResult = Association[")
            wl_content.append(f'  "point" -> {self._format_wl_array(result.best_point)},')
            wl_content.append(f'  "value" -> {format_number(result.best_value)}')
            wl_content.append("];")
        else:
            wl_content.append("bestResult = Missing[\"NotAvailable\"];")
        wl_content.append("")
        
        wl_content.append("(* History *)")
        wl_content.append("history = {")
        
        for i, ep in enumerate(history_to_export):
            is_last = (i == len(history_to_export) - 1)
            wl_content.append("  Association[")
            wl_content.append(f'    "point" -> {self._format_wl_array(ep.point)},')
            wl_content.append(f'    "value" -> {format_number(ep.value)},')
            
            ci_str = format_confidence_interval(ep.confidence_interval)
            wl_content.append(f'    "confidenceInterval" -> {ci_str},')
            
            wl_content.append(f'    "sampleCount" -> {getattr(ep, "sample_count", len(ep.raw_measurements))},')
            
            algo_str = escape_string(ep.algorithm_used)
            wl_content.append(f'    "algorithmUsed" -> {algo_str},')
            
            timestamp_str = format_timestamp(ep.timestamp)
            wl_content.append(f'    "timestamp" -> {timestamp_str}')
            
            if is_last:
                wl_content.append("  ]")
            else:
                wl_content.append("  ],")
        
        wl_content.append("};")
        wl_content.append("")
        
        wl_content.append("(* Algorithm Statistics *)")
        wl_content.append("algorithmStats = Association[")
        stats_items = list(self.algorithm_usage.items())
        for i, (algo, count) in enumerate(stats_items):
            is_last = (i == len(stats_items) - 1)
            algo_escaped = escape_string(algo)
            if is_last:
                wl_content.append(f'  {algo_escaped} -> {count}')
            else:
                wl_content.append(f'  {algo_escaped} -> {count},')
        wl_content.append("];")
        
        wl_content.append("")
        wl_content.append("(* Data Validation *)")
        wl_content.append('Print["Data Import Successful"];')
        wl_content.append('Print["Total Records: ", Length[history]];')
        wl_content.append('If[MissingQ[bestResult], Print["Best result not available"], Print["Best value: ", bestResult["value"]]];')
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(wl_content))
        
        print(f"\n 结果已导出到: {filename}")
        print(f"   格式: Wolfram Language (.wl)")
        print(f"   可在Mathematica或Wolfram语言中导入使用")

    def _format_wl_bounds(self, bounds):
        
        bounds_str = [f"{{{b[0]}, {b[1]}}}" for b in bounds]
        return "{" + ", ".join(bounds_str) + "}"
    
    def _format_wl_list(self, items):
        
        quoted_items = [f'"{item}"' for item in items]
        return "{" + ", ".join(quoted_items) + "}"
    
    def _format_wl_array(self, array):
        
        values = [str(v) for v in array]
        return "{" + ", ".join(values) + "}"

    def _calculate_parameter_sensitivity(self, result: OptimizationResult) -> Dict:
        
        sensitivity = {}

        if len(result.history) < 5:
            return sensitivity

        try:
            X = np.array([p.point for p in result.history])
            y = np.array([p.value for p in result.history])

            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LinearRegression()
            model.fit(X_scaled, y)

            for i, param_name in enumerate(self.config.param_names):
                sensitivity[param_name] = {
                    "coefficient": float(model.coef_[i]),
                    "importance": float(abs(model.coef_[i]) / np.sum(np.abs(model.coef_)) * 100)
                }

        except Exception as e:
            sensitivity["error"] = str(e)

        return sensitivity

class OptimizationProgram:

    def __init__(self):
        self.optimizer = None
        self.config = None
        self.selected_algorithm = None

    def configure(self):
        
        print(" 优化程序配置")
        print("="*50)

        n_dims = int(input("请输入参数数量 (默认2): ") or 2)

        param_names = []
        param_bounds = []

        for i in range(n_dims):
            print(f"\n参数 {i+1}:")
            name = input(f"  参数名称 (默认: 参数{i+1}): ") or f"参数{i+1}"
            param_names.append(name)

            min_val = float(input(f"  最小值 (默认: 0): ") or 0)
            max_val = float(input(f"  最大值 (默认: 100): ") or 100)
            param_bounds.append((min_val, max_val))

        num_samples = int(input(f"\n每点采样次数 (默认5): ") or 5)
        precision = int(input(f"精度截断小数位数 (默认2): ") or 2)
        max_evaluations = int(input(f"最大评估次数 (默认30): ") or 30)
        
        print(f"\n使用高级算法?")
        print("A. 全流程优化 (自动切换多种算法)")
        print("B. 拉丁方采样 (Latin Hypercube)")
        print("C. 遗传进化法 (Genetic Algorithm)")
        print("D. 共轭方向法 (Powell)")
        print("E. 粒子群算法 (Particle Swarm)")
        print("F. 模拟退火法 (Simulated Annealing)")
        print("G. 贝叶斯优化 (Bayesian Optimization)")
        
        algo_choice = input("\n请选择 (A/B/C/D/E/F/G, 默认A): ").strip().upper() or 'A'
        
        if algo_choice == 'A':
            use_advanced = True
            selected_algorithm = None
        elif algo_choice == 'B':
            use_advanced = True
            selected_algorithm = 'latin_hypercube'
        elif algo_choice == 'C':
            use_advanced = True
            selected_algorithm = 'genetic'
        elif algo_choice == 'D':
            use_advanced = True
            selected_algorithm = 'powell'
        elif algo_choice == 'E':
            use_advanced = True
            selected_algorithm = 'particle_swarm'
        elif algo_choice == 'F':
            use_advanced = True
            selected_algorithm = 'simulated_annealing'
        elif algo_choice == 'G':
            use_advanced = True
            selected_algorithm = 'bayesian_ei'
        else:
            print("  无效选择,使用默认的全流程优化")
            use_advanced = True
            selected_algorithm = None
        
        self.selected_algorithm = selected_algorithm

        self.config = OptimizationConfig(
            param_bounds=param_bounds,
            param_names=param_names,
            num_samples=num_samples,
            precision=precision,
            max_evaluations=max_evaluations,
            use_advanced_algorithms=use_advanced
        )

        print("\n 配置完成!")
        self._display_config()

    def _display_config(self):
        
        if self.config is None:
            print(" 配置未设置")
            return

        print("\n 当前配置:")
        print(f"   参数维度: {len(self.config.param_names)}")
        for name, bounds in zip(self.config.param_names, self.config.param_bounds):
            print(f"     {name}: [{bounds[0]}, {bounds[1]}]")
        print(f"   每点采样次数: {self.config.num_samples}")
        print(f"   精度: {self.config.precision} 位小数")
        print(f"   最大评估次数: {self.config.max_evaluations}")
        print(f"   使用高级算法: {'是' if self.config.use_advanced_algorithms else '否'}")

    def _get_search_directories(self):
        
        import os
        search_dirs = [
            os.getcwd(),
            os.path.expanduser("~"),
            os.path.expanduser("~/Downloads"),
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Desktop"),
        ]
        
        android_paths = [
            "/storage/emulated/0",
            "/storage/emulated/0/Download",
            "/storage/emulated/0/Documents",
            "/sdcard",
            "/sdcard/Download",
            "/sdcard/Documents",
        ]
        search_dirs.extend(android_paths)
        
        if 'EXTERNAL_STORAGE' in os.environ:
            search_dirs.append(os.environ['EXTERNAL_STORAGE'])
        
        unique_dirs = []
        seen = set()
        for d in search_dirs:
            if d not in seen and os.path.isdir(d):
                unique_dirs.append(d)
                seen.add(d)
        
        return unique_dirs
    
    def _find_file(self, filename, extensions=None):
        
        import os
        
        if os.path.isabs(filename) and os.path.isfile(filename):
            return filename
        
        if os.path.isfile(filename):
            return os.path.abspath(filename)
        
        search_dirs = self._get_search_directories()
        
        for directory in search_dirs:
            full_path = os.path.join(directory, filename)
            if os.path.isfile(full_path):
                return full_path
            
            if extensions:
                for ext in extensions:
                    if filename.endswith(ext):
                        continue
                    full_path_with_ext = os.path.join(directory, filename + ext)
                    if os.path.isfile(full_path_with_ext):
                        return full_path_with_ext
        
        return None
    
    def _show_search_paths(self, filename):
        
        import os
        search_dirs = self._get_search_directories()
        
        print("\n已尝试以下位置:")
        for i, directory in enumerate(search_dirs[:10], 1):
            full_path = os.path.join(directory, filename)
            print(f"  {i}. {full_path}")
        
        if len(search_dirs) > 10:
            print(f"  ... 以及其他 {len(search_dirs) - 10} 个位置")
        
        print("\n 提示:")
        print("  - 请确认文件确实存在")
        print("  - 可以尝试输入完整的绝对路径")
        print("  - 检查文件名拼写是否正确")

    def import_historical_data(self):
        
        print("\n 历史数据导入")
        print("="*60)
        
        print("\n 选择导入方式:")
        print("  A.  Wolfram Language (.wl) 文件导入")
        print("  B.  JSON文件批量导入 (推荐，适合10+条数据)")
        print("  C. ⌨  手动逐条输入 (适合少量数据，1-5条)")
        print("  D. ⏭  跳过导入 (done)")
        
        choice = input("\n请选择 (A/B/C/D, 默认D): ").strip().upper()
        
        if choice == '' or choice == 'D' or choice.lower() == 'done':
            print("⏭  跳过历史数据导入")
            return
        
        elif choice == 'A':
            self._import_from_wl()
        
        elif choice == 'B':
            self._import_from_json()
        
        elif choice == 'C':
            self._import_manually()
        
        else:
            print(" 无效选择，跳过导入")

    def _import_from_wl(self):
        
        print("\n Wolfram Language 文件导入")
        print("-"*60)
        
        print("\n 支持的Wolfram Language格式:")
        print("  格式1: {point -> {x1, x2, ...}, value -> y, ...}")
        print("  格式2: {{x1, x2, ..., y}, {x1, x2, ..., y}, ...}")
        print("  格式3: Association[\"point\" -> {x1, x2, ...}, \"value\" -> y, ...]")
        
        print("\n 文件路径提示:")
        print("  - 可以直接输入文件名 (如: data.wl)")
        print("  - 可以输入完整路径 (如: /storage/emulated/0/Download/data.wl)")
        print("  - 系统会自动搜索常见目录")
        
        default_filename = "optimization_history.wl"
        filename = input(f"\n WL文件名或路径 (默认: {default_filename}): ").strip() or default_filename
        
        found_path = self._find_file(filename, ['.wl', '.m', '.nb'])
        
        if found_path is None:
            print(f"\n 无法找到文件 '{filename}'")
            print("\n 已搜索以下位置:")
            self._show_search_paths(filename)
            retry = input("\n是否尝试其他文件? (y/n, 默认n): ").strip().lower()
            if retry == 'y':
                self._import_from_wl()
            return
        
        try:
            with open(found_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f" 成功读取文件: {found_path}")
            
            history_list = self._parse_wl_data(content)
            
            if history_list is None or len(history_list) == 0:
                print(f" 文件格式不支持或没有找到有效数据")
                return
            
            print(f"\n 找到 {len(history_list)} 条历史记录")
            
            imported_count = self._import_records(history_list)
            
            self._show_import_summary(imported_count, len(history_list))
        
        except Exception as e:
            print(f"\n 导入失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _parse_wl_data(self, content):
        
        import re
        
        history_list = []
        
        content = re.sub(r'\(\*.*?\*\)', '', content, flags=re.DOTALL)
        
        pattern1 = r'point\s*->\s*\{([^}]+)\}\s*,\s*value\s*->\s*([\d.+-eE]+)'
        matches1 = re.finditer(pattern1, content, re.IGNORECASE)
        for match in matches1:
            try:
                point_str = match.group(1)
                value_str = match.group(2)
                
                point = [float(x.strip()) for x in point_str.split(',')]
                value = float(value_str)
                
                history_list.append({
                    "point": point,
                    "value": value,
                    "confidence_interval": (value, value),
                    "raw_measurements": [value],
                    "algorithm_used": "wl_import"
                })
            except Exception as e:
                print(f"    解析记录失败: {e}")
        
        if len(history_list) == 0:
            pattern2 = r'\{([\d.,\s+-eE]+)\}'
            matches2 = re.finditer(pattern2, content)
            for match in matches2:
                try:
                    values_str = match.group(1)
                    values = [float(x.strip()) for x in values_str.split(',') if x.strip()]
                    
                    if len(values) >= len(self.config.param_bounds) + 1:
                        point = values[:len(self.config.param_bounds)]
                        value = values[len(self.config.param_bounds)]
                        
                        history_list.append({
                            "point": point,
                            "value": value,
                            "confidence_interval": (value, value),
                            "raw_measurements": [value],
                            "algorithm_used": "wl_import"
                        })
                except Exception as e:
                    continue
        
        if len(history_list) == 0:
            pattern3 = r'"?point"?\s*->\s*\{([^}]+)\}\s*,\s*"?value"?\s*->\s*([\d.+-eE]+)'
            matches3 = re.finditer(pattern3, content, re.IGNORECASE)
            for match in matches3:
                try:
                    point_str = match.group(1)
                    value_str = match.group(2)
                    
                    point = [float(x.strip()) for x in point_str.split(',')]
                    value = float(value_str)
                    
                    history_list.append({
                        "point": point,
                        "value": value,
                        "confidence_interval": (value, value),
                        "raw_measurements": [value],
                        "algorithm_used": "wl_import"
                    })
                except Exception as e:
                    print(f"    解析记录失败: {e}")
        
        return history_list

    def _import_from_json(self):
        
        print("\n JSON文件批量导入")
        print("-"*60)
        
        print("\n 支持的JSON格式:")
        print("  格式1 (推荐): { \"history\": [...], \"config\": {...} }")
        print("  格式2 (简单): [ {...}, {...}, ... ]")
        
        print("\n 文件路径提示:")
        print("  - 可以直接输入文件名 (如: data.json)")
        print("  - 可以输入完整路径 (如: /storage/emulated/0/Download/data.json)")
        print("  - 系统会自动搜索常见目录")
        
        default_filename = "optimization_history.json"
        filename = input(f"\n JSON文件名或路径 (默认: {default_filename}): ").strip() or default_filename
        
        found_path = self._find_file(filename, ['.json'])
        
        if found_path is None:
            print(f"\n 无法找到文件 '{filename}'")
            print("\n 已搜索以下位置:")
            self._show_search_paths(filename)
            retry = input("\n是否尝试其他文件? (y/n, 默认n): ").strip().lower()
            if retry == 'y':
                self._import_from_json()
            return
        
        try:
            with open(found_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f" 成功读取文件: {found_path}")
            
            history_list, imported_config = self._parse_json_data(data)
            
            if history_list is None:
                print(f" 文件格式不支持")
                return
            
            print(f"\n 找到 {len(history_list)} 条历史记录")
            
            if imported_config:
                self._handle_imported_config(imported_config)
            
            imported_count = self._import_records(history_list)
            
            self._show_import_summary(imported_count, len(history_list))
        
        except json.JSONDecodeError as e:
            print(f"\n 文件 '{found_path}' 不是有效的JSON格式")
            print(f"   错误详情: {e}")
        except Exception as e:
            print(f"\n 导入失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _parse_json_data(self, data):
        
        history_list = None
        config = None
        
        if isinstance(data, list):
            history_list = data
        
        elif isinstance(data, dict):
            if "history" in data:
                history_list = data["history"]
            elif "measurements" in data:
                history_list = data["measurements"]
            
            if "config" in data:
                config = data["config"]
        
        return history_list, config
    
    def _handle_imported_config(self, imported_config):
        
        print("\n  文件包含配置信息:")
        
        print(f"   当前参数数量: {len(self.config.param_names)}")
        print(f"   文件参数数量: {len(imported_config.get('param_names', []))}")
        
        if imported_config.get('param_names'):
            print(f"   文件参数名称: {', '.join(imported_config['param_names'])}")
    
    def _import_records(self, history_list):
        
        imported_count = 0
        skipped_count = 0
        
        for i, record in enumerate(history_list, 1):
            try:
                success = self._import_single_record(record, i)
                if success:
                    imported_count += 1
                else:
                    skipped_count += 1
                    
            except Exception as e:
                print(f"   记录 {i} 导入失败: {e}")
                skipped_count += 1
        
        return imported_count
    
    def _import_single_record(self, record, index):
        
        point = np.array(record.get("point", []))
        value = float(record.get("value", 0))
        confidence_interval = record.get("confidence_interval", (value, value))
        algorithm_used = record.get("algorithm_used", "historical")
        raw_measurements = record.get("raw_measurements", [value])
        
        if len(point) != len(self.config.param_bounds):
            print(f"    记录 {index}: 参数维度不匹配 ({len(point)} != {len(self.config.param_bounds)})，跳过")
            return False
        
        in_bounds = self.optimizer.is_in_current_bounds(point)
        
        eval_point = EvaluationPoint(
            point=point,
            value=value,
            confidence_interval=tuple(confidence_interval),
            raw_measurements=raw_measurements,
            timestamp=record.get("timestamp", time.time()),
            algorithm_used=algorithm_used
        )
        
        self.optimizer.all_history.append(eval_point)
        
        if in_bounds:
            self.optimizer.history.append(eval_point)
            
            if value > self.optimizer.current_best_value:
                self.optimizer.current_best_value = value
                self.optimizer.current_best = point.copy()
        else:
            if self.config.allow_out_of_bounds:
                if index <= 5 or index % 10 == 0:
                    print(f"    记录 {index}: 超出当前范围,但已保留用于模型训练")
            else:
                print(f"    记录 {index}: 参数超出范围,跳过")
                return False
        
        if value > self.optimizer.global_best_value:
            self.optimizer.global_best_value = value
            self.optimizer.global_best = point.copy()
        
        if index <= 5 or index % 10 == 0:
            formatted_point = self.optimizer._format_point(point)
            status = "" if in_bounds else " "
            print(f"  {status} 已导入记录 {index}: {formatted_point} → {value}")
        
        return True
    
    def _show_import_summary(self, imported_count, total_count):
        
        print("\n" + "="*60)
        print(f" 导入完成")
        print("="*60)
        print(f"   成功导入: {imported_count}/{total_count} 条记录")
        print(f"   跳过记录: {total_count - imported_count} 条")
        
        if imported_count > 0:
            print(f"\n 数据统计:")
            print(f"   全局历史记录: {len(self.optimizer.all_history)} 条")
            print(f"   当前范围内记录: {len(self.optimizer.history)} 条")
            
            out_of_bounds_count = len(self.optimizer.all_history) - len(self.optimizer.history)
            if out_of_bounds_count > 0:
                print(f"     超出当前范围: {out_of_bounds_count} 条 (已保留用于模型训练)")
            
            print(f"\n   当前范围最佳值: {self.optimizer.current_best_value:.{self.config.precision}f}")
            if self.optimizer.current_best is not None:
                formatted_best = self.optimizer._format_point(self.optimizer.current_best)
                print(f"   最佳参数(当前范围): {formatted_best}")
            
            if self.optimizer.global_best_value > self.optimizer.current_best_value:
                print(f"\n   ⭐ 全局最佳值: {self.optimizer.global_best_value:.{self.config.precision}f}")
                formatted_global = self.optimizer._format_point(self.optimizer.global_best)
                print(f"   最佳参数(全局): {formatted_global}")
                print(f"   注意: 全局最佳点超出当前参数范围")
    
    def _import_manually(self):
        
        print("\n⌨  手动逐条输入")
        print("-"*60)
        
        print("\n 输入格式说明:")
        print(f"   需要 {len(self.config.param_bounds)} 个参数值 + 1个测量值")
        print(f"   可选: 置信区间（下限, 上限）")
        print("\n   示例:")
        print(f"     方式1: 90, 150, 85.5              ← {len(self.config.param_bounds)}个参数 + 测量值")
        print(f"     方式2: 90, 150, 85.5, 82.3, 88.7  ← 参数 + 测量值 + 置信区间")
        print("\n   输入 'done' 或空行完成输入\n")
        
        imported_count = 0
        
        while True:
            prompt = f"记录 {imported_count + 1} (done=完成): "
            line = input(prompt).strip()
            
            if line.lower() in ['done', ''] or (line == '' and imported_count > 0):
                break
            
            try:
                success = self._parse_manual_input(line)
                if success:
                    imported_count += 1
                    print(f"   记录 {imported_count} 已添加")
            except Exception as e:
                print(f"   输入格式错误: {e}")
                print(f"   请检查格式并重新输入")
        
        if imported_count > 0:
            print(f"\n 手动导入完成: 共 {imported_count} 条记录")
            print(f"   当前历史记录总数: {len(self.optimizer.history)} 条")
        else:
            print("\n⏭  未导入任何数据")
    
    def _parse_manual_input(self, line):
        
        parts = [p.strip() for p in line.split(',')]
        values = []
        
        for p in parts:
            try:
                values.append(float(p))
            except ValueError:
                raise ValueError(f"'{p}' 不是有效数字")
        
        n_params = len(self.config.param_bounds)
        min_required = n_params + 1
        
        if len(values) < min_required:
            raise ValueError(f"需要至少 {min_required} 个值，但只提供了 {len(values)} 个")
        
        point = np.array(values[:n_params])
        
        for i, (val, bounds) in enumerate(zip(point, self.config.param_bounds)):
            if val < bounds[0] or val > bounds[1]:
                param_name = self.config.param_names[i]
                raise ValueError(
                    f"参数 '{param_name}' 值 {val} 超出范围 [{bounds[0]}, {bounds[1]}]"
                )
        
        value = values[n_params]
        
        if len(values) >= n_params + 3:
            ci_lower = values[n_params + 1]
            ci_upper = values[n_params + 2]
        else:
            ci_lower = ci_upper = value
        
        eval_point = EvaluationPoint(
            point=point,
            value=value,
            confidence_interval=(ci_lower, ci_upper),
            raw_measurements=[value],
            timestamp=time.time(),
            algorithm_used="manual"
        )
        
        self.optimizer.history.append(eval_point)
        
        if value > self.optimizer.current_best_value:
            self.optimizer.current_best_value = value
            self.optimizer.current_best = point.copy()
        
        return True

    def run(self):
        
        print("\n" + "="*80)
        print(" 增强通用优化系统 v2.0")
        print("="*80)

        self.configure()

        self.optimizer = EnhancedUniversalOptimizer(self.config)
        
        if hasattr(self, 'selected_algorithm') and self.selected_algorithm is not None:
            self.optimizer.force_algorithm = self.selected_algorithm

        import_choice = input("\n是否导入历史数据? (y/n, 默认n): ").strip().lower()
        if import_choice == 'y':
            self.import_historical_data()
        else:
            print("⏭  跳过历史数据导入")

        result = self.optimizer.run_optimization()

        self.optimizer.analyze_results(result)

        export_choice = input("\n是否导出结果? (y/n, 默认y): ").strip().lower()
        if export_choice != 'n':
            print("\n 导出结果")
            print("="*60)
            print("\n 选择导出格式:")
            print("  A.  Wolfram Language (.wl) 格式")
            print("  B.  JSON (.json) 格式 (推荐)")
            print("  C. ⏭  取消导出")
            
            format_choice = input("\n请选择格式 (A/B/C, 默认B): ").strip().upper()
            
            if format_choice == 'C':
                print("⏭  取消导出")
            else:
                print("\n 选择导出范围:")
                print("  A. 所有已知历史数据 (包括超出当前范围的)")
                print("  B. 当前参数范围内的数据")
                
                range_choice = input("\n请选择导出范围 (A/B, 默认B): ").strip().upper()
                export_all = (range_choice == 'A')
                
                if export_all:
                    print(f"\n 将导出所有历史数据 ({len(self.optimizer.all_history)}条)")
                else:
                    print(f"\n 将导出当前范围内数据 ({len(self.optimizer.history)}条)")
                
                if format_choice == 'A':
                    filename = input("\n文件名 (默认: optimization_results.wl): ") or "optimization_results.wl"
                    self.optimizer.export_results_wl(result, filename, export_all=export_all)
                else:
                    filename = input("\n文件名 (默认: optimization_results.json): ") or "optimization_results.json"
                    self.optimizer.export_results(result, filename, export_all=export_all)
        else:
            print("⏭  跳过结果导出")

        print("\n" + "="*80)
        print(" 优化完成!")
        print("="*80)

        return result

    def export_data(self):
        
        print("\n 导出历史数据")
        print("="*60)
        print("\n 选择导出格式:")
        print("  A.  Wolfram Language (.wl) 格式")
        print("  B.  JSON (.json) 格式 (推荐)")
        print("  C. ⏭  取消导出")
        
        export_choice = input("\n请选择格式 (A/B/C, 默认B): ").strip().upper()
        
        if export_choice == 'C':
            print("⏭  取消导出")
            return
        
        print("\n 选择导出范围:")
        print("  A. 所有已知历史数据 (包括超出当前范围的)")
        print("  B. 当前参数范围内的数据")
        
        range_choice = input("\n请选择导出范围 (A/B, 默认B): ").strip().upper()
        export_all = (range_choice == 'A')
        
        if export_all:
            print(f"\n 将导出所有历史数据 ({len(self.optimizer.all_history)}条)")
        else:
            print(f"\n 将导出当前范围内数据 ({len(self.optimizer.history)}条)")
        
        temp_result = OptimizationResult(
            best_point=self.optimizer.current_best,
            best_value=self.optimizer.current_best_value,
            history=self.optimizer.history,
            config=self.config,
            total_time=0.0,
            iterations=len(self.optimizer.history)
        )
        
        if export_choice == 'A':
            filename = input("\n文件名 (默认: exported_data.wl): ") or "exported_data.wl"
            self.optimizer.export_results_wl(temp_result, filename, export_all=export_all)
        else:
            filename = input("\n文件名 (默认: exported_data.json): ") or "exported_data.json"
            self.optimizer.export_results(temp_result, filename, export_all=export_all)

def main():
    
    print(" 增强通用优化系统")
    print("版本: 2.0.4")
    print("作者: AI Assistant")
    print("\n功能:")
    print("  1. 自定义参数优化")
    print("  2. 多种高级算法")
    print("  3. 多格式数据导入导出")
    print("  4. 实时数据分析")

    program = OptimizationProgram()

    while True:
        print("\n" + "="*60)
        print("主菜单")
        print("="*60)
        print("A. 完整优化流程")
        print("B. 导入历史数据")
        print("C. 导出历史数据")
        print("D. 仅配置不运行")
        print("E. 退出")

        choice = input("\n请选择 (A/B/C/D/E): ").strip().upper()

        if choice == 'A':
            program.run()

        elif choice == 'B':
            if not hasattr(program, 'config') or program.config is None:
                print("\n  请先配置参数!")
                program.configure()
            
            if not hasattr(program, 'optimizer') or program.optimizer is None:
                program.optimizer = EnhancedUniversalOptimizer(program.config)
            
            program.import_historical_data()

        elif choice == 'C':
            if not hasattr(program, 'optimizer') or program.optimizer is None:
                print("\n  没有可导出的数据!")
                print("   请先运行优化或导入历史数据")
            elif len(program.optimizer.history) == 0:
                print("\n  历史记录为空,无数据可导出!")
            else:
                program.export_data()

        elif choice == 'D':
            program.configure()

        elif choice == 'E':
            print("\n 感谢使用，再见!")
            break

        else:
            print(" 无效选择，请重试")

    input("\n按 Enter 键退出...")

if __name__ == "__main__":

    main()