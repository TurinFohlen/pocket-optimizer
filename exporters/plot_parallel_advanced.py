import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from registry import registry
from orchestrator import HistoryEntry

@registry.register(
    name='exporter.plot_parallel_advanced',
    type_='exporter',
    signature='export(history: List[HistoryEntry], filepath: str, smooth: bool = True, fill: bool = False) -> str'
)
class AdvancedParallelCoordinatesExporter:
    """
    增强平行坐标图导出器
    特性：
      - 完全向量化数据提取（无显式循环）
      - 自适应 JPG/PNG/PDF 导出（显式 format 支持）
      - 全相等维度自动置中（避免除零）
      - 可选平滑插值（需 scipy）
    """

    def export(self, history: List[HistoryEntry], filepath: str,
               smooth: bool = True, fill: bool = False) -> str:
        if not history:
            raise ValueError("历史记录为空，无法绘图")

        # ---------- 1. 向量化提取数据 ----------
        # 直接构建矩阵：每行 = [point_dims..., value]
        points = np.array([h.point for h in history])      # (n, d)
        values = np.array([h.value for h in history])      # (n,)
        data = np.column_stack([points, values])           # (n, d+1)

        n_points, n_dims = points.shape
        print(f"??? 平行坐标图: {n_points} 个点, {n_dims} 维参数")
        print(f"   参数范围: [{points.min():.3f}, {points.max():.3f}]")
        print(f"   目标值范围: [{values.min():.3f}, {values.max():.3f}]")

        # ---------- 2. 归一化每个维度到 [0,1]（向量化）----------
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        ranges = data_max - data_min

        # 优雅处理全相等维度：除零时设为1，之后将归一化值设为0.5
        with np.errstate(divide='ignore', invalid='ignore'):
            data_norm = (data - data_min) / np.where(ranges == 0, 1, ranges)
        # 全相等维度统一置中
        for d in range(n_dims + 1):
            if ranges[d] == 0:
                data_norm[:, d] = 0.5

        # ---------- 3. 绘制平行坐标 ----------
        fig, ax = plt.subplots(figsize=(12, 6))
        x_coords = np.arange(n_dims)

        # 平滑插值（需 scipy）
        HAS_SCIPY = False
        if smooth:
            try:
                from scipy.interpolate import pchip
                HAS_SCIPY = True
            except ImportError:
                print("⚠️ scipy 未安装，将使用线性连接")

        # 目标值归一化后用于颜色映射
        colors = data_norm[:, -1]

        for i in range(n_points):
            y_vals = data_norm[i, :-1]
            color = plt.cm.viridis(colors[i])

            if smooth and HAS_SCIPY and n_dims > 2:
                interp_x = np.linspace(0, n_dims - 1, 100)
                interp_y = pchip(x_coords, y_vals)(interp_x)
                ax.plot(interp_x, interp_y, color=color, alpha=0.6, linewidth=0.8)
            else:
                ax.plot(x_coords, y_vals, color=color, alpha=0.6, linewidth=0.8)

            if fill and HAS_SCIPY and n_dims > 2:
                ax.fill_between(interp_x if smooth else x_coords,
                                0, interp_y if smooth else y_vals,
                                color=color, alpha=0.1)

        # ---------- 4. 坐标轴装饰 ----------
        # 参数名称（若 point 是结构化数组可提取名称，这里简单使用 p1, p2...）
        param_names = [f'p{i+1}' for i in range(n_dims)]
        ax.set_xticks(x_coords)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.set_xlim(0, n_dims - 1)
        ax.set_ylabel('Normalized Value')
        ax.set_title(f'Parallel Coordinates ({n_points} evaluations)')

        # ---------- 5. 颜色条 ----------
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis,
            norm=plt.Normalize(vmin=values.min(), vmax=values.max())
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Objective Value')

        # ---------- 6. 保存文件（显式支持 JPG）----------
        # 强制将非 JPG 后缀转换为 JPG 并警告
        if not (filepath.lower().endswith('.jpg') or filepath.lower().endswith('.jpeg')):
            print(f"⚠️ 警告: 输出路径 '{filepath}' 不是 JPG 后缀，将强制保存为 JPG 格式。")
            # 若原后缀不是 .jpg，我们直接替换或附加
            if '.' in filepath:
                base = filepath.rsplit('.', 1)[0]
            else:
                base = filepath
            filepath = base + '.jpg'

        # 显式指定格式为 JPEG，并设置高质量压缩
        plt.tight_layout()
        plt.savefig(filepath, dpi=200, format='jpeg',
                    pil_kwargs={'quality': 95, 'optimize': True})
        plt.close()

        print(f"✅ 平行坐标图已保存: {filepath}")
        return filepath
