import json
from typing import List
from registry import registry
from orchestrator import HistoryEntry

@registry.register(
    name='exporter.json',
    type_='exporter',
    signature='export(history: List[HistoryEntry], filepath: str) -> str'
)
class JSONExporter:
    def export(self, history: List[HistoryEntry], filepath: str) -> str:
        data = []
        for h in history:
            data.append({
                'timestamp': h.timestamp,
                'algorithm': h.algorithm,
                'point': h.point.tolist(),
                'value': h.value
            })
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return filepath