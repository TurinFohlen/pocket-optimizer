import numpy as np
from registry import registry


@registry.register(
    name='source.test_rugged_rotated',
    type_='source',
    signature='measure(point: np.ndarray, n_samples: int) -> float'
)
class TestRuggedRotatedSource:
    """
    高难度旋转多峰函数（最大化问题）

    特点：
    - 非凸
    - 多局部极值
    - 旋转破坏坐标结构
    - 峡谷结构
    - 轻微噪声
    """

    def __init__(self):
        self.measurement_count = 0
        self.noise_level = 0.01
        self.max_evals = 1000
    def measure(self, point: np.ndarray, n_samples: int = 5) -> float:
        self.measurement_count += 1

        true_value = self._rugged_function(point)

        measurements = []
        for _ in range(n_samples):
            noise = np.random.normal(0, self.noise_level)
            measurements.append(true_value + noise)

        return float(np.mean(measurements))

    def _rugged_function(self, point: np.ndarray) -> float:
        point = np.atleast_1d(point)

        if len(point) < 2:
            x = point[0]
            y = 0.0
        else:
            x, y = point[0], point[1]

        # 旋转
        theta = np.pi / 5
        c, s = np.cos(theta), np.sin(theta)

        u = c * x - s * y
        v = s * x + c * y

        # 主峡谷（非对称）
        quad = -(0.5 * u**2 + 5.0 * v**2)

        # 多峰干扰
        ripple = 2.0 * np.sin(3*u) * np.sin(3*v)
        ripple += 1.5 * np.exp(-0.1*(u**2 + v**2)) * np.cos(8*u) * np.cos(8*v)

        # 欺骗性局部极值
        trap = -0.8 * np.exp(-((u - 2.5)**2 + (v + 2.0)**2))

        value = quad + ripple + trap

        return float(value)