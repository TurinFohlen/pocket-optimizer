# services/data/history_adapter_service.py
import numpy as np
from registry import registry


@registry.register(
    name="service.data.history_adapter",
    type_="service",
    signature="convert(data: list) -> dict"
)
class HistoryAdapterService:

    def convert(self, data):
        """
        将 HistoryEntry 列表转换为：
        {
            "positions": np.ndarray (N, D),
            "values":    np.ndarray (N,)
        }
        """

        if not data:
            return {
                "positions": np.empty((0, 0)),
                "values": np.empty((0,))
            }

        positions = []
        values = []

        for record in data:
            positions.append(record.point)   # ← 关键修正
            values.append(record.value)

        return {
            "positions": np.asarray(positions, dtype=float),
            "values": np.asarray(values, dtype=float),
        }