import loader
from typing import List
from registry import registry
from orchestrator import HistoryEntry

@registry.register(
    name='exporter.excel',
    type_='exporter',
    signature='export(history: List[HistoryEntry], filepath: str) -> str'
)
class ExcelExporter:
    def export(self, history: List[HistoryEntry], filepath: str) -> str:
        try:
            from openpyxl import Workbook
        except ImportError:
            raise ImportError("openpyxl not installed. Run: pip install openpyxl")
        
        wb = Workbook()
        ws = wb.active
        ws.append(['timestamp', 'algorithm', 'point', 'value'])
        for h in history:
            ws.append([
                h.timestamp,
                h.algorithm,
                str(h.point.tolist()),
                h.value
            ])
        wb.save(filepath)
        return filepath