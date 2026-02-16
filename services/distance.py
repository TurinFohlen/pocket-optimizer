# services/distance.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array
from registry import registry

@registry.register(name='service.distance', type_='service')
class DistanceService:
    """工业级距离服务（单例）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._points = np.empty((0, 0))   # 历史点矩阵
            self._nn = {}                     # 不同度量的索引树缓存
            self._max_samples = 10000
            self._initialized = True
    
    # ------------------------------------------------------------------
    # 核心：构建索引树（每个度量独立缓存）
    # ------------------------------------------------------------------
    def _get_tree(self, metric: str, **kwargs) -> NearestNeighbors:
        """获取或构建指定度量的最近邻树"""
        if len(self._points) < 2:
            return None
        
        # 标准化参数作为缓存键
        if metric == 'minkowski':
            p = kwargs.get('p', 2)
            key = f'minkowski_p{p}'
        else:
            key = metric
        
        if key not in self._nn:
            # 使用 ball_tree 对切比雪夫/曼哈顿/欧氏均高效
            self._nn[key] = NearestNeighbors(
                n_neighbors=min(5, len(self._points)),
                metric=metric,
                algorithm='ball_tree',
                **kwargs
            ).fit(self._points)
        return self._nn[key]
    
    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    def add_points(self, X: np.ndarray):
        """批量添加点（自动维护历史矩阵）"""
        X = check_array(X, ensure_2d=True)
        if self._points.size == 0:
            self._points = X
        else:
            self._points = np.vstack([self._points, X])
        # 限制历史长度，防止内存爆炸
        if len(self._points) > self._max_samples:
            self._points = self._points[-self._max_samples:]
        # 索引树失效，清空缓存
        self._nn.clear()
    
    def kneighbors(self, X: np.ndarray, n_neighbors: int = 5,
                   metric: str = 'euclidean', **kwargs):
        """
        最近邻查询
        - 度量支持: 'euclidean', 'manhattan', 'chebyshev', 'minkowski'
        - 返回值: (距离数组, 索引数组)
        """
        if len(self._points) == 0:
            return np.array([]), np.array([])
        
        X = check_array(X, ensure_2d=True)
        n_neighbors = min(n_neighbors, len(self._points))
        
        tree = self._get_tree(metric, **kwargs)
        if tree is None:
            # 完全无依赖回退：用 scipy.spatial.distance.cdist（更优）
            from scipy.spatial.distance import cdist
            dists = cdist(X, self._points, metric=metric, **kwargs)
            idx = np.argsort(dists, axis=1)[:, :n_neighbors]
            dists = np.take_along_axis(dists, idx, axis=1)
            return dists, idx
        else:
            # sklearn 树查询
            dists, idx = tree.kneighbors(X, n_neighbors=n_neighbors)
            return dists, idx
    
    # ------------------------------------------------------------------
    # 单点便捷方法
    # ------------------------------------------------------------------
    def add_point(self, point: np.ndarray):
        """添加单个点"""
        self.add_points(point.reshape(1, -1))
    
    def distance(self, a: np.ndarray, b: np.ndarray, metric: str = 'euclidean', **kwargs) -> float:
        """直接计算两点距离（委托给 scipy.spatial.distance）"""
        from scipy.spatial.distance import cdist
        return float(cdist(a.reshape(1, -1), b.reshape(1, -1), metric=metric, **kwargs)[0, 0])
    
    @property
    def size(self) -> int:
        """当前存储点数"""
        return len(self._points)


# 全局单例实例
distance = DistanceService()