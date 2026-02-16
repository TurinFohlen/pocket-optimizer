import numpy as np
import matplotlib.pyplot as plt
from typing import List
from registry import registry
from orchestrator import HistoryEntry


@registry.register(
    name='exporter.convergence_plot',
    type_='exporter',
    signature='export(history: List[HistoryEntry], filepath: str) -> str'
)
class ConvergencePlotExporter:
    def export(self, history: List[HistoryEntry], filepath: str) -> str:
        if not history:
            print("Warning: No history data to plot")
            return ""
        
        values = np.array([h.value for h in history])
        iterations = np.arange(1, len(values) + 1)
        
        best_so_far = np.maximum.accumulate(values)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(iterations, values, 'o-', alpha=0.6, markersize=3, linewidth=1, label='Measured Value')
        ax1.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best So Far')
        ax1.set_xlabel('Evaluation #', fontsize=12)
        ax1.set_ylabel('Objective Value', fontsize=12)
        ax1.set_title(f'Convergence History ({len(history)} evaluations)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        window_size = max(10, len(values) // 50)
        if len(values) >= window_size:
            moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(range(window_size, len(values)+1), moving_avg, 'g-', linewidth=2, label=f'Moving Average (window={window_size})')
            ax2.plot(iterations, values, 'b.', alpha=0.2, markersize=2, label='Raw Values')
            ax2.set_xlabel('Evaluation #', fontsize=12)
            ax2.set_ylabel('Objective Value', fontsize=12)
            ax2.set_title('Smoothed Trend', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.plot(iterations, values, 'bo-', markersize=4)
            ax2.set_xlabel('Evaluation #', fontsize=12)
            ax2.set_ylabel('Objective Value', fontsize=12)
            ax2.set_title('All Evaluations', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Convergence plot saved: {filepath}")
        print(f"  Total evaluations: {len(history)}")
        print(f"  Best value: {max(values):.6f}")
        print(f"  Final value: {values[-1]:.6f}")
        
        return filepath
