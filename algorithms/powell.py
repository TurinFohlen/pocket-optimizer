import numpy as np
from typing import List, Tuple
from registry import registry

# ========== 服务基建懒加载 ==========
_svc = None
def _get_dist_svc():
    global _svc
    if _svc is None:
        svc_cls = registry.get_component('service.distance')
        _svc = svc_cls() if svc_cls else None
    return _svc

_exec = None
def _get_exec_svc():
    global _exec
    if _exec is None:
        svc_cls = registry.get_component('service.executor')
        _exec = svc_cls() if svc_cls else None
    return _exec


@registry.register(
    name='algorithm.powell',
    type_='algorithm',
    signature='optimize(bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]'
)
class PowellAlgorithm:
    required_source = 'source.interactive'

    def __init__(self, source):
        self.source = source
        self.max_iterations = 20
        self.tolerance = 1e-6
        self.n_samples = 5

        # ========== 看门狗参数（可调）==========
        self.boundary_threshold = 0.1          # 距离边界 10% 内视为“边界区域”
        self.compression_factor = 0.5          # 反射幅度压缩系数
        self.duplicate_threshold = 0.01        # 切比雪夫距离 < 1% 范围视为重复点
        self.far_enough_threshold = 0.3        # 回退模式中步长 > 30% 切比雪夫距离才尝试

    def optimize(self, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        n_dims = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        span = upper - lower

        # 初始点（中心）
        x0 = np.array([(b[0] + b[1]) / 2.0 for b in bounds])

        # 获取服务
        dist_svc = _get_dist_svc()
        exec_svc = _get_exec_svc()

        # ========== 带看门狗的反射边界映射 ==========
        def reflect_to_bounds(y: np.ndarray) -> np.ndarray:
            """
            压缩反射：当点在边界附近时，减小反射跳跃幅度
            """
            x = np.zeros_like(y)
            for i in range(n_dims):
                l, h = bounds[i][0], bounds[i][1]
                span_i = h - l
                if span_i <= 0:
                    x[i] = l
                    continue

                # 看门狗①：边界距离检测（切比雪夫单维化）
                dist_to_lower = abs(y[i] - l)
                dist_to_upper = abs(y[i] - h)
                near_boundary = min(dist_to_lower, dist_to_upper) < self.boundary_threshold * span_i

                # 标准化位置 [0,2)
                t = ((y[i] - l) / span_i) % 2.0

                # 看门狗②：反射幅度压缩
                if near_boundary:
                    if t < 1.0:
                        t = 1.0 - (1.0 - t) * self.compression_factor
                    else:
                        t = 1.0 + (t - 1.0) * self.compression_factor

                # 反射映射
                if t < 1.0:
                    x[i] = l + t * span_i
                else:
                    x[i] = l + (2.0 - t) * span_i
            return x

        # ========== SciPy Powell 主路径（带看门狗③：重复点抑制）==========
        try:
            from scipy.optimize import minimize

            # 评估缓存（同时用于重复点跳过）
            evaluated_points = {}

            def objective_for_scipy(y: np.ndarray) -> float:
                # 反射到可行域
                x = reflect_to_bounds(y)

                # 看门狗③：重复点抑制（基于切比雪夫距离）
                if dist_svc and dist_svc.size > 10:
                    dists, _ = dist_svc.kneighbors(x.reshape(1, -1), n_neighbors=1, metric='chebyshev')
                    if len(dists) > 0 and dists[0] < self.duplicate_threshold * np.max(span):
                        x_tuple = tuple(np.round(x, decimals=10))
                        # 若已有缓存，直接返回负值（最大化问题）
                        if x_tuple in evaluated_points:
                            return -evaluated_points[x_tuple]
                        else:
                            # 全新重复点：给一个极差的值，迫使优化器离开
                            return -1e10

                # 正常评估
                x_tuple = tuple(np.round(x, decimals=10))
                if x_tuple in evaluated_points:
                    return -evaluated_points[x_tuple]

                value = self.source.measure(x, n_samples=self.n_samples)
                evaluated_points[x_tuple] = value

                # 记录历史点（用于后续重复检测）
                if dist_svc:
                    dist_svc.add_point(x)

                return -value

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
            x_tuple = tuple(np.round(best_x, decimals=10))
            best_value = evaluated_points.get(x_tuple,
                                              self.source.measure(best_x, n_samples=self.n_samples))
            return best_x, best_value

        except ImportError:
            # ========== 无 SciPy 回退模式 ==========
            print("SciPy 未安装，使用回退 Powell 搜索（带并发优化）")
            best_point = x0.copy()
            best_value = self.source.measure(best_point, n_samples=self.n_samples)

            # 初始化距离服务（若无则跳过）
            if dist_svc:
                dist_svc.add_point(best_point)

            # 定义单方向测试函数
            def test_direction(dim, direction_sign):
                test_point = best_point.copy()
                step = (bounds[dim][1] - bounds[dim][0]) * 0.1
                test_val = best_point[dim] + direction_sign * step
                if bounds[dim][0] <= test_val <= bounds[dim][1]:
                    test_point[dim] = test_val
                    # 重复点抑制
                    if dist_svc and dist_svc.size > 5:
                        dists, _ = dist_svc.kneighbors(test_point.reshape(1, -1), n_neighbors=1, metric='chebyshev')
                        if len(dists) > 0 and dists[0] < self.duplicate_threshold * np.max(span):
                            return None  # 跳过
                    val = self.source.measure(test_point, n_samples=self.n_samples)
                    return (test_point, val)
                return None

            # 生成所有候选方向
            candidates = []
            for dim in range(n_dims):
                for direction in [1, -1]:
                    candidates.append((dim, direction))

            # 并发评估（若执行器服务可用）
            if exec_svc:
                results = exec_svc.map(
                    lambda args: test_direction(*args),
                    candidates,
                    pool_type='thread',
                    max_workers=4
                )
            else:
                results = [test_direction(d, s) for d, s in candidates]

            # 处理结果
            for res in results:
                if res is not None:
                    test_point, test_value = res
                    if test_value > best_value:
                        best_value = test_value
                        best_point = test_point.copy()
                        if dist_svc:
                            dist_svc.add_point(best_point)

            return best_point, best_value