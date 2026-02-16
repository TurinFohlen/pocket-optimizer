"""
绘图导出器 —— 将优化历史绘制为收敛曲线
依赖 matplotlib (pip install matplotlib)
"""

from typing import List
from registry import registry
from orchestrator import HistoryEntry

@registry.register(
    name='exporter.plot',
    type_='exporter',
    signature='export(history: List[HistoryEntry], filepath: str) -> str'
)
class PlotExporter:
    def export(self, history: List[HistoryEntry], filepath: str) -> str:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib not installed. Run: pip install matplotlib"
            )
        
        # 提取数据
        timestamps = [h.timestamp for h in history]
        values = [h.value for h in history]
        algorithms = [h.algorithm for h in history]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制收敛曲线
        ax.plot(range(len(values)), values, 'b-o', markersize=4, label='Best Value')
        
        # 标注最终最优值
        best_idx = values.index(max(values))
        ax.plot(best_idx, values[best_idx], 'r*', markersize=12, 
                label=f'Optimum: {values[best_idx]:.4f}')
        
        # 装饰
        ax.set_xlabel('Evaluation Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('Optimization Convergence History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        
        return filepath