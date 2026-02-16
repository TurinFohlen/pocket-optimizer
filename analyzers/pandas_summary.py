from registry import registry

@registry.register(
    name='exporter.excel_pandas',
    type_='exporter',
    signature='export(history: List[HistoryEntry], filepath: str) -> str'
)
class PandasExcelExporter:
    def export(self, history, filepath):
        try:
            import pandas as pd
        except ImportError:
            return self._fallback_excel(history, filepath)
        
        df = pd.DataFrame([{
            'timestamp': h.timestamp,
            'algorithm': h.algorithm,
            'value': h.value,
            'point': str(h.point.tolist())
        } for h in history])
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='History', index=False)
            df.describe().to_excel(writer, sheet_name='Summary')
        return filepath
    
    def _fallback_excel(self, history, filepath):
        """无 pandas 时回退到基础 Excel 导出"""
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.append(['timestamp', 'algorithm', 'point', 'value'])
        for h in history:
            ws.append([h.timestamp, h.algorithm, str(h.point.tolist()), h.value])
        wb.save(filepath)
        return filepath