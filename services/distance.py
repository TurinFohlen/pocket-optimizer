# services/distance.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array
from registry import registry


@registry.register(
    name="processor.distance",
    type_="processor",
    signature="distance(a: np.ndarray, b: np.ndarray) -> float"
)
class DistanceService:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._points = np.empty((0, 0))
            self._nn = {}
            self._max_samples = 10000
            self._initialized = True

    # -------------------------------
    # 索引树
    # -------------------------------
    def _get_tree(self, metric: str, **kwargs) -> NearestNeighbors:
        if len(self._points) < 2:
            return None

        if metric == 'minkowski':
            p = kwargs.get('p', 2)
            key = f'minkowski_p{p}'
        else:
            key = metric

        if key not in self._nn:
            self._nn[key] = NearestNeighbors(
                n_neighbors=min(5, len(self._points)),
                metric=metric,
                algorithm='ball_tree',
                **kwargs
            ).fit(self._points)

        return self._nn[key]

    # -------------------------------
    # 公共 API
    # -------------------------------
    def add_points(self, X: np.ndarray):
        X = check_array(X, ensure_2d=True)

        if self._points.size == 0:
            self._points = X
        else:
            self._points = np.vstack([self._points, X])

        if len(self._points) > self._max_samples:
            self._points = self._points[-self._max_samples:]

        self._nn.clear()

    def kneighbors(self, X: np.ndarray, n_neighbors: int = 5,
                   metric: str = 'euclidean', **kwargs):

        if len(self._points) == 0:
            return np.array([]), np.array([])

        X = check_array(X, ensure_2d=True)
        n_neighbors = min(n_neighbors, len(self._points))

        tree = self._get_tree(metric, **kwargs)

        if tree is None:
            from scipy.spatial.distance import cdist
            dists = cdist(X, self._points, metric=metric, **kwargs)
            idx = np.argsort(dists, axis=1)[:, :n_neighbors]
            dists = np.take_along_axis(dists, idx, axis=1)
            return dists, idx
        else:
            return tree.kneighbors(X, n_neighbors=n_neighbors)

    def add_point(self, point: np.ndarray):
        self.add_points(point.reshape(1, -1))

    def distance(self, a: np.ndarray, b: np.ndarray,
                 metric: str = 'euclidean', **kwargs) -> float:
        from scipy.spatial.distance import cdist
        return float(
            cdist(a.reshape(1, -1), b.reshape(1, -1),
                  metric=metric, **kwargs)[0, 0]
        )

    @property
    def size(self) -> int:
        return len(self._points)


# 注册后获取单例实例（可选）
distance = DistanceService()