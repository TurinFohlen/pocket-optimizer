import importlib.util, os
from registry import registry
from orchestrator import Orchestrator, OptimizationConfig


def _load_adapter():
    """加载 HistoryAdapterService（文件名含点，需手动加载）"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, 'services', 'data.history_adapter_service.py')
    spec = importlib.util.spec_from_file_location('history_adapter', path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HistoryAdapterService()


def _call_exporter(exporter, exp_name, history, filepath):
    if 'lagrangian' in exp_name:
        try:
            adapter = _load_adapter()
        except Exception as e:
            print(f"HistoryAdapterService 加载失败: {e}")
            return
        data = adapter.convert(history)
        if data['positions'].shape[0] < 4:
            print("历史点数不足（需要 >= 4），无法生成拉格朗日景观图")
            return
        paths = exporter.export(data, filepath=filepath)
        for p in paths:
            print(f"  -> {p}")
    else:
        exporter.export(history, filepath)


@registry.register(
    name='ui.cli_menu',
    type_='ui',
    signature='run()'
)
class ClassicCLI:
    def run(self):
        print("\n=== 优化系统 · 经典菜单 ===\n")

        # 1. 选择测量源
        sources = registry.list_components('source')
        if not sources:
            print("没有可用的测量源")
            return
        print("可用测量源：")
        for i, s in enumerate(sources):
            print(f"  {i}. {s.name}")
        src_idx = int(input("请选择 [0]: ") or "0")
        source_name = sources[src_idx].name

        # 2. 配置优化问题
        dims = int(input("维度 [2]: ") or "2")
        bounds, param_names = [], []
        for i in range(dims):
            low  = float(input(f"  参数{i+1}下限 [-5]: ") or "-5")
            high = float(input(f"  参数{i+1}上限 [5]: ")  or "5")
            bounds.append((low, high))
            param_names.append(f"x{i+1}")

        # 采样次数
        n_samples = int(input("每点采样次数 [5]: ") or "5")

        # 优化方向
        print("优化方向：")
        print("  1. 最大化（找最大值）")
        print("  2. 最小化（找最小值）")
        dir_raw = input("请选择 [1]: ").strip() or "1"
        maximize = (dir_raw != "2")
        dir_label = "最大化" if maximize else "最小化"

        config = OptimizationConfig(
            param_bounds=bounds,
            param_names=param_names,
            num_samples=n_samples,
            maximize=maximize,
        )

        # 3. 选择算法
        algos = registry.list_components('algorithm')
        if not algos:
            print("没有可用的算法")
            return
        print("\n可用算法：")
        for i, a in enumerate(algos):
            print(f"  {i}. {a.name}")
        algo_idx = int(input("请选择 [0]: ") or "0")
        algo_name = algos[algo_idx].name

        # 4. 执行优化
        print(f"\n正在运行 {algo_name}  [{dir_label}  采样x{n_samples}] ...")
        orch = Orchestrator(config, source_name=source_name)
        best, val = orch.run(algo_name)
        print(f"\n最优解: {best}")
        print(f"最优值: {val:.6f}  ({dir_label})")

        # 5. 导出结果
        exporters = registry.list_components('exporter')
        if not exporters:
            return

        print("\n导出结果：")
        for i, e in enumerate(exporters):
            print(f"  {i+1}. {e.name}")
        raw = input("选择导出格式 [不导出]: ").strip()
        if not raw:
            return

        exp_idx = int(raw) - 1
        if not (0 <= exp_idx < len(exporters)):
            print("跳过导出")
            return

        exp_name = exporters[exp_idx].name
        exp_cls  = registry.get_component(exp_name)
        exporter = exp_cls()

        algo_short = algo_name.split('.')[-1]
        if 'lagrangian' in exp_name:
            filepath = f"lle_{algo_short}"
        elif 'plot' in exp_name or 'convergence' in exp_name:
            filepath = f"result_{algo_short}.png"
        elif 'csv' in exp_name:
            filepath = f"result_{algo_short}.csv"
        elif 'excel' in exp_name or 'pandas' in exp_name:
            filepath = f"result_{algo_short}.xlsx"
        elif 'json' in exp_name:
            filepath = f"result_{algo_short}.json"
        else:
            filepath = f"result_{algo_short}.out"

        history = orch.get_history()
        _call_exporter(exporter, exp_name, history, filepath)
        print("导出完成")

        # 历史摘要
        print(f"\n历史记录: {len(history)} 次评估  [{dir_label}  采样x{n_samples}]")
        if history:
            best_h = max(history, key=lambda h: h.value) if maximize \
                     else min(history, key=lambda h: h.value)
            print(f"  首次: {history[0].value:.4f}")
            print(f"  最优: {best_h.value:.4f}  @ {best_h.point}")
            print(f"  末次: {history[-1].value:.4f}")
