import csv
from registry import registry

@registry.register(
    name='exporter.excel_pandas',
    type_='exporter',
    signature='export(history: List[HistoryEntry], filepath: str) -> str'
)
class PandasExcelExporter:
    def export(self, history, filepath):
        # 优先尝试 pandas + openpyxl
        try:
            import pandas as pd
            import openpyxl  # 显式检测，缺失时直接跳到 except
            df = self._to_df(pd, history)
            if not filepath.endswith('.xlsx'):
                filepath = filepath.rsplit('.', 1)[0] + '.xlsx'
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='History', index=False)
                df.describe().to_excel(writer, sheet_name='Summary')
            print(f"  [xlsx] 已用 pandas+openpyxl 导出")
            return filepath

        except ImportError as e:
            missing = str(e).split("'")[1] if "'" in str(e) else str(e)
            print(f"  ⚠️  {missing} 未安装，回退到 CSV 导出")
            return self._fallback_csv(history, filepath)

    def _to_df(self, pd, history):
        return pd.DataFrame([{
            'iteration': h.iteration,
            'algorithm': h.algorithm,
            'value':     h.value,
            'point':     str(h.point.tolist()),
            'timestamp': h.timestamp,
        } for h in history])

    def _fallback_csv(self, history, filepath):
        """无 pandas/openpyxl 时回退 CSV，安卓 Pydroid 环境兼容"""
        if not filepath.endswith('.csv'):
            filepath = filepath.rsplit('.', 1)[0] + '.csv'
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'algorithm', 'value', 'point', 'timestamp'])
            for h in history:
                writer.writerow([h.iteration, h.algorithm, h.value,
                                 str(h.point.tolist()), h.timestamp])
        print(f"  [csv] 已回退 CSV 导出（无需额外依赖）")
        return filepath
