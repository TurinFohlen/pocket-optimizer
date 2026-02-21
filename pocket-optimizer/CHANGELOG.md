# Changelog

## [1.0.0-stable] - 2026-02-18

### 结构收敛重构（内核稳定版）

本版本不增加新功能，专注于将系统从"能用"提升到"可长期演进"的内核稳定态。

---

### Phase 1 · 接口纯化

**目标**：算法层只做搜索，Source 层只做测量，采样策略完全内聚到 Source。

- `measure(point)` 接口统一纯化，移除所有 Source 的 `n_samples` 参数
  - `source.interactive`
  - `source.interactive_classic`
  - `source.test_function`
  - `source.test_cli`
  - `source.test_rugged_rotated`
- `n_samples` 内聚到各 Source 的 `__init__`，调用方通过构造函数配置采样策略
- 5 个算法移除 `self.n_samples` 属性及透传调用
  - `algorithm.pso`
  - `algorithm.simulated_annealing`
  - `algorithm.bayesian`
  - `algorithm.powell`
  - `algorithm.genetic`
- `Orchestrator.SourceWrapper.measure` 签名同步纯化
- `LagrangianLandscapeExporter.export()` 移除运行时 `registry.get_service` 依赖，
  改为直接接收物理字典 `{"positions": (N,D), "values": (N,)}`；
  `HistoryAdapterService` 作为外部桥接层，不内嵌于 exporter
- `Registry._wrap_callable` 补加 `functools.wraps`，
  确保方法包装后 `inspect.signature` 返回原始签名

---

### Phase 2 · 职责边界确认

**目标**：消除重复记录，明确各层历史归属。

确认并固化以下职责边界（已符合，无需改动）：

| 历史类型 | 归属 | 说明 |
|---|---|---|
| 评估日志（point + value + 时间戳） | `Orchestrator.SourceWrapper` | 统一写入，算法不重复 |
| 算法内部状态（PSO 最优轨迹等） | Algorithm | 驱动搜索逻辑，业务合理 |
| 测量历史（噪声统计、LOF 数据） | Source | 测量策略的一部分，业务合理 |

---

### Phase 3 · 版本冻结

- 创建 `VERSION` 文件，标记内核稳定版 `1.0.0-stable`
- 创建本 `CHANGELOG.md`

---

### 扩展指南

后续在此版本基础上添加功能时，请遵守：

1. **新 Source**：实现 `measure(self, point: np.ndarray) -> float`，采样次数通过 `__init__` 配置
2. **新 Algorithm**：只依赖 `source.measure(point)`，不保存通用评估日志，不持有 `n_samples`
3. **新 Exporter**：接收 `List[HistoryEntry]`（标准路径）或物理字典（图形学路径），不做数据计算
4. **新图形学 Exporter**：接口定义为 `export(data: {"positions": (N,D), "values": (N,)}, ...)`，
   需要 HistoryEntry 时通过 `HistoryAdapterService` 在调用方转换
5. **Registry 使用原则**：`@registry.register` 用于声明，`registry.get_service` 只在显式依赖注入场景使用，不在核心方法体内隐式查找
