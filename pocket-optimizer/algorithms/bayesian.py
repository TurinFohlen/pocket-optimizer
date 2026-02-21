import numpy as np
from typing import List, Tuple, Callable
from registry import registry

# 服务懒加载
_dist_svc = None
def _get_dist_svc():
    global _dist_svc
    if _dist_svc is None:
        svc_cls = registry.get_component('service.distance')
        _dist_svc = svc_cls() if svc_cls else None
    return _dist_svc

_exec_svc = None
def _get_exec_svc():
    global _exec_svc
    if _exec_svc is None:
        svc_cls = registry.get_component('service.executor')
        _exec_svc = svc_cls() if svc_cls else None
    return _exec_svc


@registry.register(
    name='algorithm.bayesian',
    type_='algorithm',
    signature='optimize(bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]'
)
class BayesianOptimization:
    required_source = 'source.interactive'

    def __init__(self, source):
        self.source = source
        self.n_initial = 5
        self.n_iterations = 25
        self.xi = 0.01
        self.n_restarts = 10
        self.candidate_ratio = 100
        self.duplicate_threshold_initial = 0.618
        self.duplicate_decay = 0.1
        self.batch_size = 50

        self.dist_svc = _get_dist_svc()
        self.exec_svc = _get_exec_svc()

    def optimize(self, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        """贝叶斯优化主循环（增强版：切比雪夫去重 + 并发EI）"""
        n_dims = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        span = np.max(upper - lower)

        X = []
        y = []
        for _ in range(self.n_initial):
            x = np.random.uniform(lower, upper, size=n_dims)
            val = self.source.measure(x)
            X.append(x)
            y.append(val)
            if self.dist_svc:
                self.dist_svc.add_point(x)

        X = np.array(X)
        y = np.array(y)

        best_idx = np.argmax(y)
        best_x = X[best_idx].copy()
        best_y = y[best_idx]

        for it in range(self.n_iterations):
            mu_func, sigma_func = self._gaussian_process(X, y, bounds)

            n_candidates = n_dims * self.candidate_ratio
            X_cand = self._lhs_sample(lower, upper, n_candidates)

            # 切比雪夫去重：过滤掉离历史点太近的候选点
            if self.dist_svc and self.dist_svc.size > 5:
                threshold = self.duplicate_threshold_initial * span * np.exp(-self.duplicate_decay * len(X))
                dists, _ = self.dist_svc.kneighbors(X_cand, n_neighbors=1, metric='chebyshev')
                keep = dists[:, 0] >= threshold
                X_cand = X_cand[keep]
                if len(X_cand) == 0:
                    X_cand = self._lhs_sample(lower, upper, 50)

            if len(X_cand) == 0:
                x_next = np.random.uniform(lower, upper, size=n_dims)
            else:
                f_best = y.max()
                if self.exec_svc and len(X_cand) > self.batch_size:
                    def process_batch(batch):
                        mu_batch, sigma_batch = self._predict_batch(mu_func, sigma_func, batch, X, bounds)
                        scores = self._expected_improvement_batch(mu_batch, sigma_batch, f_best, self.xi)
                        best_in_batch = batch[np.argmax(scores)]
                        return best_in_batch, np.max(scores)

                    batches = [X_cand[i:i+self.batch_size] for i in range(0, len(X_cand), self.batch_size)]
                    results = self.exec_svc.map(process_batch, batches, pool_type='thread', max_workers=4)
                    candidates_from_batches = np.array([r[0] for r in results])
                    scores_from_batches = np.array([r[1] for r in results])
                    x_next = candidates_from_batches[np.argmax(scores_from_batches)]
                else:
                    mu_cand, sigma_cand = self._predict_batch(mu_func, sigma_func, X_cand, X, bounds)
                    scores = self._expected_improvement_batch(mu_cand, sigma_cand, f_best, self.xi)
                    x_next = X_cand[np.argmax(scores)]

            y_next = self.source.measure(x_next)

            X = np.vstack([X, x_next])
            y = np.append(y, y_next)
            if self.dist_svc:
                self.dist_svc.add_point(x_next)

            if y_next > best_y:
                best_y = y_next
                best_x = x_next.copy()

        return best_x, best_y

    def _gaussian_process(self, X_train: np.ndarray, y_train: np.ndarray,
                         bounds: List[Tuple[float, float]]) -> Tuple[Callable, Callable]:
        n_dims = len(bounds)
        length_scale = 0.1
        signal_variance = 1.0
        noise_variance = 0.01

        def kernel(x1, x2):
            dist = np.sum((x1 - x2) ** 2)
            return signal_variance * np.exp(-dist / (2 * length_scale ** 2))

        K = np.zeros((len(X_train), len(X_train)))
        for i in range(len(X_train)):
            for j in range(len(X_train)):
                K[i, j] = kernel(X_train[i], X_train[j])
        K += noise_variance * np.eye(len(X_train))

        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        except np.linalg.LinAlgError:
            K += 1e-6 * np.eye(len(X_train))
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

        def mu(x):
            k_x = np.array([kernel(x, x_i) for x_i in X_train])
            return np.dot(k_x, alpha)

        def sigma(x):
            k_x = np.array([kernel(x, x_i) for x_i in X_train])
            v = np.linalg.solve(L, k_x)
            var = kernel(x, x) - np.dot(v, v)
            return np.sqrt(max(var, 1e-10))

        return mu, sigma

    def _predict_batch(self, mu: Callable, sigma: Callable,
                       X: np.ndarray, X_train: np.ndarray,
                       bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        n = len(X)
        mu_vals = np.zeros(n)
        sigma_vals = np.zeros(n)
        for i in range(n):
            mu_vals[i] = mu(X[i])
            sigma_vals[i] = sigma(X[i])
        return mu_vals, sigma_vals

    def _expected_improvement_batch(self, mu: np.ndarray, sigma: np.ndarray,
                                    f_best: float, xi: float) -> np.ndarray:
        from scipy import stats
        imp = mu - f_best - xi
        z = imp / (sigma + 1e-9)
        ei = imp * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        ei[sigma < 1e-10] = 0.0
        return ei

    def _lhs_sample(self, lower: np.ndarray, upper: np.ndarray, n: int) -> np.ndarray:
        n_dims = len(lower)
        samples = np.random.uniform(0, 1, size=(n, n_dims))
        for d in range(n_dims):
            intervals = np.linspace(0, 1, n + 1)
            samples[:, d] = np.random.uniform(intervals[:-1], intervals[1:])
            np.random.shuffle(samples[:, d])
            samples[:, d] = lower[d] + samples[:, d] * (upper[d] - lower[d])
        return samples

    # 兼容原版方法（保持接口一致）
    def _acquisition_optimize(self, mu: Callable, sigma: Callable,
                            f_best: float, bounds: List[Tuple[float, float]]) -> np.ndarray:
        return np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])

    def _norm_pdf(self, x: float) -> float:
        from scipy import stats
        return stats.norm.pdf(x)

    def _norm_cdf(self, x: float) -> float:
        from scipy import stats
        return stats.norm.cdf(x)
