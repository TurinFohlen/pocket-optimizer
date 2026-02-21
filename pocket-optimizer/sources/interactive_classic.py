"""
极致精简版·原味交互测量源
依赖: numpy, scipy (可选), sklearn (可选, 用于LOF增强)
"""

import numpy as np
import time
from typing import List, Tuple, Optional
from registry import registry

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.neighbors import LocalOutlierFactor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@registry.register(
    name='source.interactive_classic',
    type_='source',
    signature='measure(point: np.ndarray) -> float'
)
class InteractiveClassicSource:
    """精简版·原味交互测量源 - 依赖库函数，代码量减少70%"""
    
    def __init__(self, n_samples: int = 5):
        self.measurement_history = []  # 用于LOF增强（业务需要）
        self.n_samples = n_samples

    def measure(self, point: np.ndarray) -> float:
        # ---------- 1. 友好的参数显示 ----------
        print(f"\n📍 测量点: {self._format_point(point)}")
        print(f"   采样次数: {self.n_samples}")

        # ---------- 2. 逐次输入测量值 ----------
        raw_values = []
        for i in range(self.n_samples):
            while True:
                try:
                    val = float(input(f"   第 {i+1}/{self.n_samples} 次测量值: "))
                    raw_values.append(val)
                    break
                except ValueError:
                    print("   错误: 请输入有效数字")

        values = np.array(raw_values)

        # ---------- 3. 多重异常值检测（自适应样本量）----------
        filtered = self._filter_outliers(values, point)
        n_orig, n_filt = len(values), len(filtered)
        if n_filt < n_orig:
            print(f"   过滤掉 {n_orig - n_filt} 个异常值")

        # ---------- 4. 稳健均值估计（修整均值）----------
        mean_val = self._robust_mean(filtered)
        mean_val = round(mean_val, 6)

        # ---------- 5. 置信区间（t分布 / bootstrap）----------
        if n_filt >= 2:
            ci = self._confidence_interval(filtered)
            ci_low, ci_high = round(ci[0], 6), round(ci[1], 6)
            width = ci_high - ci_low
            rel_width = width / (abs(mean_val) + 1e-10) * 100
            print(f"   95% 置信区间: [{ci_low:.6f}, {ci_high:.6f}]")
            print(f"   区间宽度: {width:.6f} ({rel_width:.1f}%)")
        else:
            print(f"   有效样本不足 ({n_filt})，置信区间不可用")

        # ---------- 6. 记录历史（用于LOF增强）----------
        self.measurement_history.append({
            'point': point.copy(),
            'values': raw_values.copy(),
            'mean': mean_val,
            'timestamp': time.time()
        })
        if len(self.measurement_history) > 100:
            self.measurement_history.pop(0)

        print(f"   稳健均值: {mean_val:.6f}")
        return float(mean_val)

    # ------------------------------------------------------------------
    # 以下为精简后的核心方法，每个方法1-5行，完全依赖成熟库函数
    # ------------------------------------------------------------------

    def _filter_outliers(self, values: np.ndarray, point: np.ndarray) -> np.ndarray:
        """自适应异常值检测 - 组合IQR, Z-Score, LOF增强"""
        n = len(values)
        
        # 1. 极少量样本：LOF增强（若可用）
        if n <= 4 and HAS_SKLEARN and len(self.measurement_history) >= 3:
            return self._lof_augmented_filter(values, point)
        
        # 2. 通用检测：IQR + Z-Score
        mask = np.ones(n, dtype=bool)
        
        # IQR 方法（对非正态分布稳健）
        q1, q3 = np.percentile(values, [25, 75])
        iqr = max(q3 - q1, 1e-10)
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        mask &= (values >= lower_iqr) & (values <= upper_iqr)
        
        # Z-Score 方法（正态假设）
        if HAS_SCIPY and n >= 8:  # 样本量足够时才用
            z_scores = np.abs(stats.zscore(values, ddof=1))
            mask &= (z_scores < 3)
        else:
            # 稳健Z-Score（基于MAD）
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            if mad > 0:
                robust_z = 0.6745 * (values - median) / mad
                mask &= (np.abs(robust_z) < 3.5)
        
        # 3. 百分位数截断（保留95%）
        lower_pct = np.percentile(values, 5)
        upper_pct = np.percentile(values, 95)
        mask &= (values >= lower_pct) & (values <= upper_pct)
        
        filtered = values[mask]
        
        # 4. 过度过滤保护 - 至少保留2个样本
        if len(filtered) < 2:
            median = np.median(values)
            distances = np.abs(values - median)
            idx = np.argsort(distances)[:2]  # 保留离中位数最近的两个
            filtered = values[idx]
            print(f"   恢复离中位数最近的 {len(filtered)} 个样本")
        
        return filtered

    def _lof_augmented_filter(self, values: np.ndarray, point: np.ndarray) -> np.ndarray:
        """LOF增强检测 - 利用历史数据"""
        try:
            # 构建特征矩阵：当前值 + 最近10个历史均值
            X = np.array([[v, 0] for v in values])
            for hist in self.measurement_history[-10:]:
                X = np.vstack([X, [hist['mean'], 1]])
            
            lof = LocalOutlierFactor(n_neighbors=min(10, len(X)-1), contamination=0.1)
            y_pred = lof.fit_predict(X)
            
            # 返回未标记为离群点的当前值
            inliers = [values[i] for i in range(len(values)) if y_pred[i] == 1]
            return np.array(inliers) if inliers else values
        except Exception:
            return values

    def _robust_mean(self, values: np.ndarray) -> float:
        """稳健均值 - 使用修整均值（trimmed mean）"""
        if HAS_SCIPY:
            # 自适应修整比例：样本越少，修整越少
            trim = min(0.1, 0.5 / len(values)) if len(values) > 2 else 0
            return float(stats.trim_mean(values, trim))
        else:
            # 回退：去掉最大最小值后平均
            if len(values) >= 4:
                return float(np.mean(np.sort(values)[1:-1]))
            return float(np.mean(values))

    def _confidence_interval(self, values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """置信区间 - 小样本t分布，大样本正态近似，中等样本bootstrap"""
        n = len(values)
        if n < 2:
            m = np.mean(values) if n == 1 else 0.0
            return (m, m)
        
        if not HAS_SCIPY:
            # 无scipy：百分位数法
            return tuple(np.percentile(values, [(1-confidence)/2*100, (1+confidence)/2*100]))
        
        # 1. t分布（小样本最优）
        mean = np.mean(values)
        se = np.std(values, ddof=1) / np.sqrt(n)
        ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=se)
        
        # 2. 中等样本用bootstrap验证
        if 8 <= n < 50:
            try:
                # scipy 1.8+ 有 bootstrap
                from scipy.stats import bootstrap
                res = bootstrap((values,), np.mean, confidence_level=confidence, 
                                n_resamples=1000, method='BCa')
                boot_ci = res.confidence_interval
                # 若bootstrap区间与t区间差异>50%，采用bootstrap
                if abs((boot_ci[1]-boot_ci[0]) - (ci[1]-ci[0])) / (ci[1]-ci[0] + 1e-10) > 0.5:
                    ci = (boot_ci[0], boot_ci[1])
            except (ImportError, AttributeError):
                pass
        
        return ci

    def _format_point(self, point: np.ndarray) -> str:
        """简洁的参数格式化"""
        return ", ".join(f"p{i+1}={v:.6f}" for i, v in enumerate(point))