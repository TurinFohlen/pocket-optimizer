import csv
import loader
from typing import List
from registry import registry
from orchestrator import HistoryEntry

@registry.register(
    name='exporter.csv',
    type_='exporter',
    signature='export(history: List[HistoryEntry], filepath: str) -> str'
)
class CSVExporter:
    def export(self, history: List[HistoryEntry], filepath: str) -> str:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'algorithm', 'point', 'value'])
            for h in history:
                writer.writerow([
                    h.timestamp,
                    h.algorithm,
                    h.point.tolist(),
                    h.value
                ])
        return filepath