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