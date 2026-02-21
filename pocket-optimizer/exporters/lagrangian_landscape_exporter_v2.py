from __future__ import annotations

import os
import itertools
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata

from matplotlib.colors import hsv_to_rgb

from registry import registry
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'sans-serif',    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],    'font.size': 11,    'axes.titlesize': 12,    'axes.labelsize': 11,    'xtick.labelsize': 10,    'ytick.labelsize': 10,    'legend.fontsize': 10,    'figure.dpi': 300,    'savefig.dpi': 300,    'axes.linewidth': 0.8,    'xtick.direction': 'in',    'ytick.direction': 'in',    'xtick.minor.visible': True,    'ytick.minor.visible': True,    'legend.frameon': False, 'lines.linewidth': 1.0, "axes.unicode_minus": False
    })
#统一字体、字号、刻度方向，出版级基础





"""
Lagrangian Landscape Exporter (LLE)
可视化模型：LLE(F(x,y,ρ)) ——将搜索算法历史轨迹映射为拉格朗日景观图

数学模型
--------
T(x)  : Chebyshev 迭代动能  E_N_sigma(x)[d_inf(x_{t+1}, x_t)^2]
V(x)  : 高斯平滑核势能       E_N_sigma(x)[phi(d(x, x_i))],  phi(r)=1/(r+eps)
L(x)  : 拉格朗日密度          T(x) + V(x)
|gF|  : 目标函数梯度幅值
rho   : 采样密度              gaussian_kde

可视化映射（多组合输出）
------------------------
等高线   <- F（固定）
Hue / Saturation / Value <- {L, |gF|, rho} 的全排列组合
按结构重要性升序排列，导出末尾 K 张（最重要的 K 张）

输入接口（仅 numpy.ndarray）
----------------------------
{
    "positions": np.ndarray (N, D),   仅使用前两维作为坐标轴
    "values":    np.ndarray (N,)
}
"""


# ─────────────────────────────────────────────────────
# 纯函数层（无副作用，无 I/O）
# ─────────────────────────────────────────────────────

def _gaussian_weights(center: np.ndarray,
                      points: np.ndarray,
                      sigma: float) -> np.ndarray:
    sq = np.sum((points - center) ** 2, axis=1)
    w = np.exp(-sq / (2.0 * sigma ** 2))
    s = w.sum()
    return w / s if s > 0.0 else np.full(len(w), 1.0 / max(len(w), 1))


def _kinetic_energy(positions: np.ndarray, sigma: float) -> np.ndarray:
    """T(x_i) = E_{N_sigma(x_i)}[d_inf(x_{t+1}, x_t)^2]"""
    N = len(positions)
    if N < 2:
        return np.zeros(N)
    steps = positions[1:] - positions[:-1]
    cheby_sq = np.max(np.abs(steps), axis=1) ** 2
    midpoints = 0.5 * (positions[:-1] + positions[1:])
    T = np.empty(N)
    for i in range(N):
        w = _gaussian_weights(positions[i], midpoints, sigma)
        T[i] = float(w @ cheby_sq)
    return T


def _potential_energy(positions: np.ndarray,
                      sigma: float,
                      epsilon: float) -> np.ndarray:
    """V(x_i) = E_{N_sigma(x_i)}[phi(d(x_i, x_j))],  phi(r) = 1/(r+eps)"""
    dists = cdist(positions, positions, metric='euclidean')
    phi = 1.0 / (dists + epsilon)
    np.fill_diagonal(phi, 0.0)
    V = np.empty(len(positions))
    for i in range(len(positions)):
        w = _gaussian_weights(positions[i], positions, sigma)
        V[i] = float(w @ phi[i])
    return V


def _to_grid(pos2d: np.ndarray, field: np.ndarray,
             gx: np.ndarray, gy: np.ndarray,
             smooth: float) -> np.ndarray:
    """散点场 -> 规则网格（linear 插值 + nearest 补边 + 高斯平滑）。"""
    z = griddata(pos2d, field, (gx, gy), method='linear')
    mask = np.isnan(z)
    if mask.any():
        z[mask] = griddata(pos2d, field, (gx, gy), method='nearest')[mask]
    return gaussian_filter(z, sigma=smooth)


def _grad_mag(grid: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(grid)
    return np.sqrt(gx ** 2 + gy ** 2)


def _kde_grid(pos2d: np.ndarray,
              gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    kde = gaussian_kde(pos2d.T)
    pts = np.vstack([gx.ravel(), gy.ravel()])
    return kde(pts).reshape(gx.shape)


def _norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-12)


# ── 各通道的感知优化处理策略 ────────────────────────────────────

def _to_hue(arr: np.ndarray) -> np.ndarray:
    """
    -> Hue: 映射到 0.05-0.72，避开红色双端折叠。
    颜色变化丰富，不丢失区分度。
    """
    return 0.05 + 0.67 * _norm(arr)


def _to_saturation(arr: np.ndarray) -> np.ndarray:
    """
    -> Saturation: 保底 0.50，防止低值区退化为白色/灰色。
    sqrt 拉伸确保低值区也有足够饱和度（解决过曝核心问题）。
    """
    return 0.50 + 0.50 * np.sqrt(_norm(arr))


def _to_value(arr: np.ndarray) -> np.ndarray:
    """
    -> Value (Lightness): log 压缩峰值，范围锁定 [0.28, 0.80]。
    高密度区不再过亮；低密度区保持可见亮度。
    """
    n = _norm(arr)
    compressed = np.log1p(5.0 * n) / np.log1p(5.0)
    return 0.28 + 0.52 * compressed


_CHANNEL_FN = {'H': _to_hue, 'S': _to_saturation, 'V': _to_value}


# ── 组合评分 ──────────────────────────────────────────────────────

def _importance_score(grids: List[np.ndarray],
                      channels: Tuple[str, ...]) -> float:
    """
    组合重要性 = 独立性 x 信息量（各通道方差加权）。
    通道感知权重：H > S > V（人眼对色调最敏感）。
    """
    weight = {'H': 1.0, 'S': 0.75, 'V': 0.50}
    info = sum(g.var() * weight[c] for g, c in zip(grids, channels))

    M = np.column_stack([g.ravel() for g in grids])
    M_std = (M - M.mean(0)) / (M.std(0) + 1e-12)
    corr = np.corrcoef(M_std.T)
    rows, cols = np.triu_indices(len(grids), k=1)
    indep = float(1.0 - np.abs(corr[rows, cols]).mean()) if len(rows) else 1.0

    return indep * info


def _build_combinations(var_grids: Dict[str, np.ndarray],
                        K: int) -> List[Dict]:
    """
    {L, |gF|, rho} -> {H, S, V} 全排列（3! = 6 种），
    按重要性升序排列，返回末尾 K 个（最重要）。
    """
    names = list(var_grids.keys())
    channels = ('H', 'S', 'V')
    combos = []
    for perm in itertools.permutations(names):
        grids = [var_grids[v] for v in perm]
        score = _importance_score(grids, channels)
        combos.append({
            'mapping': dict(zip(channels, perm)),
            'grids':   dict(zip(channels, grids)),
            'score':   score,
        })
    combos.sort(key=lambda c: c['score'])
    return combos[-min(K, len(combos)):]


# ─────────────────────────────────────────────────────
# 单图绘制
# ─────────────────────────────────────────────────────

_BG = '#0a0a0a'


def _render_one(combo: Dict,
                F_grid: np.ndarray,
                grid_x: np.ndarray,
                grid_y: np.ndarray,
                pos2d: np.ndarray,
                extent: Tuple[float, float, float, float],
                meta: str,
                rank: int,
                total: int,
                filepath: str) -> str:
    mapping  = combo['mapping']
    grids_ch = combo['grids']

    H_arr = _CHANNEL_FN['H'](grids_ch['H'])
    S_arr = _CHANNEL_FN['S'](grids_ch['S'])
    V_arr = _CHANNEL_FN['V'](grids_ch['V'])

    rgb = hsv_to_rgb(np.dstack([H_arr, S_arr, V_arr]))

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    ax.imshow(rgb, origin='lower', extent=extent,
              aspect='auto', interpolation='bicubic')

    _f_range = F_grid.max() - F_grid.min()
    if _f_range < 1e-10:
        # F 近似常数（如测试函数退化），跳过等高线
        cs = None
    else:
        levels = np.linspace(F_grid.min(), F_grid.max(), 18)
        cs = ax.contour(grid_x, grid_y, F_grid,
                    levels=levels, colors='white',
                    linewidths=0.65, alpha=0.50
                    )
    if cs is not None:
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f', colors='white')

    ax.scatter(pos2d[:, 0], pos2d[:, 1],
               c='white', s=6, alpha=0.30, linewidths=0, zorder=3
               )
    ax.scatter(*pos2d[0],  c='lime', s=55, marker='^', zorder=5, label='start')
    ax.scatter(*pos2d[-1], c='red',  s=55, marker='*', zorder=5, label='end')

    mapping_str = '  |  '.join(f'{ch}={v}' for ch, v in mapping.items())
    score_str = f"score={combo['score']:.4f}"
    ax.text(0.02, 0.02,
            f"[{rank}/{total}]  {mapping_str}\n{meta}  {score_str}",
            transform=ax.transAxes, fontsize=7, color='white', va='bottom',
            bbox=dict(facecolor='black', alpha=0.50, pad=3))

    hue_var = mapping['H']
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv,
                               norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.01)
    cbar.set_label(f'Hue = {hue_var}', fontsize=10, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    cbar.ax.yaxis.set_tick_params(color='white')

    ax.set_xlabel('x0', fontsize=9, color='white')
    ax.set_ylabel('x1', fontsize=9, color='white')
    ax.set_title(
        f'LLE  —  Combination {rank}/{total}  '
        f'(score {combo["score"]:.3f})\n'
        f'Contour=F  |  {mapping_str}',
        fontsize=9, color='white', pad=8
    )
    ax.legend(fontsize=7, loc='upper right',
              facecolor='black', labelcolor='white', framealpha=0.5)
    ax.tick_params(
    axis='both',          # x 和 y 轴都应用
    colors='white',       # 刻度线颜色
    labelcolor='white'    # 刻度标签（数字）颜色 ← 关键！原来缺这个
    )
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')

    plt.tight_layout()
    plt.savefig(filepath.replace('.png', '.pdf'),
            bbox_inches='tight',
            format='pdf',
            dpi=300)
    plt.close(fig)
    return filepath


# ─────────────────────────────────────────────────────
# 主导出器
# ─────────────────────────────────────────────────────

@registry.register(
    name='exporter.lagrangian_landscape_v2',
    type_='exporter',
    signature=(
        'export(data: dict, filepath: str, '
        'sigma: float, epsilon: float, K: int, grid_n: int) -> list'
    )
)
class LagrangianLandscapeExporter:
    """
    拉格朗日景观导出器 (LLE)

    输出 K 张图，每张对应 {L, |gF|, rho} 映射到 {H, S, V} 的一种排列组合。
    等高线固定为 F；坐标轴固定为 positions 的前两个维度（D>=2 时忽略剩余维度）。
    按结构重要性升序排列，导出末尾 K 张（最重要的 K 张）。
    """

    def export(
        self,
        data: Dict[str, np.ndarray],
        filepath: str = "",
        sigma: float = 0.5,
        epsilon: float = 1e-6,
        K: int = 6,
        grid_n: int = 128,
    ) -> List[str]:
        """
        参数
        ----
        data     : {"positions": (N,D), "values": (N,)}
                   D >= 2；可视化仅使用 positions[:,0] 和 positions[:,1]
        filepath : 输出路径前缀（留空时自动生成）
                   每张图追加 _combo{01..K}.png
        sigma    : 高斯邻域核宽
        epsilon  : 势能奇点防护值 eps > 0
        K        : 导出图张数（<= 7，全排列最多 6 张）
        grid_n   : 可视化网格分辨率

        返回
        ----
        List[str]：生成的文件路径列表，按重要性升序（最后一个最重要）
        """
        positions = np.asarray(data["positions"], dtype=float)
        values    = np.asarray(data["values"],    dtype=float)

        if positions.ndim != 2 or positions.shape[1] < 2:
            raise ValueError("positions 必须是 (N, D)，D >= 2")
        if values.ndim != 1 or len(values) != len(positions):
            raise ValueError("values 必须是长度 N 的一维数组")

        K = max(1, min(K, 7))
        pos2d = positions[:, :2]      # 仅取前两维用于网格与可视化

        # ── 1. 物理量计算（使用全维度 positions）─────────────
        T     = _kinetic_energy(positions, sigma)
        V_pts = _potential_energy(positions, sigma, epsilon)
        L_pts = T + V_pts

        # ── 2. 网格构建 ────────────────────────────────────────
        x0, x1 = pos2d[:, 0].min(), pos2d[:, 0].max()
        y0, y1 = pos2d[:, 1].min(), pos2d[:, 1].max()
        px = (x1 - x0) * 0.05 + 1e-6
        py = (y1 - y0) * 0.05 + 1e-6
        gx_lin = np.linspace(x0 - px, x1 + px, grid_n)
        gy_lin = np.linspace(y0 - py, y1 + py, grid_n)
        grid_x, grid_y = np.meshgrid(gx_lin, gy_lin)
        extent = (x0 - px, x1 + px, y0 - py, y1 + py)

        smooth = max(0.3, sigma)

        # ── 3. 插值到网格 ──────────────────────────────────────
        F_grid    = _to_grid(pos2d, values,  grid_x, grid_y, smooth)
        L_grid    = _to_grid(pos2d, L_pts,   grid_x, grid_y, smooth)
        grad_grid = _grad_mag(F_grid)
        rho_grid  = _kde_grid(pos2d, grid_x, grid_y)

        # ── 4. 组合生成与重要性排序 ────────────────────────────
        var_grids = {'L': L_grid, '|gF|': grad_grid, 'rho': rho_grid}
        combos = _build_combinations(var_grids, K)

        # ── 5. 输出路径前缀 ────────────────────────────────────
        if filepath:
            base = os.path.splitext(filepath)[0]
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            base = f"lle_{ts}"

        # ── 6. 逐组合绘图输出 ──────────────────────────────────
        N = len(positions)
        D = positions.shape[1]
        meta = f"N={N}  D={D}  sigma={sigma}  eps={epsilon}"
        total = len(combos)
        out_paths = []

        for rank, combo in enumerate(combos, start=1):
            fpath = f"{base}_combo{rank:02d}.pdf"
            _render_one(
                combo=combo,
                F_grid=F_grid,
                grid_x=grid_x,
                grid_y=grid_y,
                pos2d=pos2d,
                extent=extent,
                meta=meta,
                rank=rank,
                total=total,
                filepath=fpath,
            )
            out_paths.append(fpath)

        return out_paths
