import numpy as np
import matplotlib.pyplot as plt
from typing import List
from registry import registry
from orchestrator import HistoryEntry
from datetime import datetime
import os


@registry.register(
    name='exporter.covergence_ln_plot',
    type_='exporter',
    signature='export(history: List[HistoryEntry], filepath: str) -> str'
)
class ConvergenceLnPlotExporter:

    def export(self, history: List[HistoryEntry], filepath: str) -> str:
        if not history:
            print("Warning: No history data to plot")
            return ""

        raw_values = np.array([h.value for h in history])
        iterations = np.arange(1, len(raw_values) + 1)

        # ---------- 数值稳定 ----------
        epsilon = 1e-12
        abs_values = np.clip(np.abs(raw_values), epsilon, None)
        values = np.log(abs_values)

        # 防止 nan / inf
        values = np.nan_to_num(values, nan=0.0, neginf=-50, posinf=50)

        # ---------- 最大化逻辑 ----------
        best_raw = np.maximum.accumulate(raw_values)
        best_abs = np.clip(np.abs(best_raw), epsilon, None)
        best_ln = np.log(best_abs)

        # ---------- 画图 ----------
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 上图：真实 ln|value|
        ax1.plot(iterations, values, 'o-', alpha=0.6,
                 markersize=3, linewidth=1,
                 label='ln|Measured Value|')

        ax1.plot(iterations, best_ln, 'r-',
                 linewidth=2,
                 label='Best ln|Value| So Far')

        ax1.set_xlabel('Evaluation #')
        ax1.set_ylabel('ln|Objective Value|')
        ax1.set_title(
            f'Convergence History (Log-Absolute, {len(history)} evaluations)',
            fontweight='bold'
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 下图：平滑趋势
        window_size = max(10, len(values) // 50)

        if len(values) >= window_size:
            kernel = np.ones(window_size) / window_size
            moving_avg = np.convolve(values, kernel, mode='valid')

            ax2.plot(range(window_size, len(values) + 1),
                     moving_avg, 'g-',
                     linewidth=2,
                     label=f'Moving Average (window={window_size})')

            ax2.plot(iterations, values,
                     'b.', alpha=0.2,
                     markersize=2,
                     label='Raw ln|Values|')

            ax2.set_title('Smoothed Trend (Log-Absolute)',
                          fontweight='bold')
        else:
            ax2.plot(iterations, values, 'bo-', markersize=4)
            ax2.set_title('All Evaluations (Log-Absolute)',
                          fontweight='bold')

        ax2.set_xlabel('Evaluation #')
        ax2.set_ylabel('ln|Objective Value|')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        base, ext = os.path.splitext(filepath)
        filepath = f"{base}_{timestamp}{ext}"

        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Log-absolute convergence plot saved: {filepath}")
        print(f"  Total evaluations: {len(history)}")
        print(f"  Best raw value: {np.max(raw_values):.6f}")
        print(f"  Final raw value: {raw_values[-1]:.6f}")

        return filepath