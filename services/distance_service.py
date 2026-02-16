import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array
from registry import registry

@registry.register(
    name='service.distance',
    type_='service',
    signature='distance_service()'
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

    def add_points(self, X: np.ndarray):
        X = check_array(X, ensure_2d=True)
        if self._points.size == 0:
            self._points = X
        else:
            self._points = np.vstack([self._points, X])
        if len(self._points) > self._max_samples:
            self._points = self._points[-self._max_samples:]
        self._nn.clear()

    def add_point(self, point: np.ndarray):
        self.add_points(point.reshape(1, -1))

    def _get_tree(self, metric: str, **kwargs):
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

    def kneighbors(self, X: np.ndarray, n_neighbors: int = 5,
                   metric: str = 'euclidean', **kwargs):
        if len(self._points) == 0:
            return np.array([]), np.array([])
        X = check_array(X, ensure_2d=True)
        n_neighbors = min(n_neighbors, len(self._points))
        tree = self._get_tree(metric, **kwargs)
        if tree is None:
            try:
                from scipy.spatial.distance import cdist
                dists = cdist(X, self._points, metric=metric, **kwargs)
            except ImportError:
                # 纯 numpy 回退
                dists = np.zeros((len(X), len(self._points)))
                for i, x in enumerate(X):
                    for j, p in enumerate(self._points):
                        if metric == 'chebyshev':
                            dists[i,j] = np.max(np.abs(x - p))
                        elif metric == 'euclidean':
                            dists[i,j] = np.sqrt(np.sum((x - p)**2))
                        else:
                            dists[i,j] = np.sum(np.abs(x - p))
            idx = np.argsort(dists, axis=1)[:, :n_neighbors]
            dists = np.take_along_axis(dists, idx, axis=1)
            return dists, idx
        else:
            return tree.kneighbors(X, n_neighbors=n_neighbors)

    @property
    def size(self):
        return len(self._points)
