from registry import registry
from orchestrator import Orchestrator, OptimizationConfig

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
            print("❌ 没有可用的测量源")
            return
        print("可用测量源：")
        for i, s in enumerate(sources):
            print(f"  {i}. {s.name}")
        src_idx = int(input("请选择 [0]: ") or "0")
        source_name = sources[src_idx].name

        # 2. 配置优化问题
        dims = int(input("维度 [2]: ") or "2")
        bounds = []
        param_names = []
        for i in range(dims):
            low = float(input(f"参数{i+1}下限 [-5]: ") or "-5")
            high = float(input(f"参数{i+1}上限 [5]: ") or "5")
            bounds.append((low, high))
            param_names.append(f"x{i+1}")

        config = OptimizationConfig(
            param_bounds=bounds,
            param_names=param_names,
            num_samples=3
        )

        # 3. 选择算法
        algos = registry.list_components('algorithm')
        if not algos:
            print("❌ 没有可用的算法")
            return
        print("\n可用算法：")
        for i, a in enumerate(algos):
            print(f"  {i}. {a.name}")
        algo_idx = int(input("请选择 [0]: ") or "0")
        algo_name = algos[algo_idx].name

        # 4. 执行优化
        print(f"\n??? 正在运行 {algo_name} ...")
        orch = Orchestrator(config, source_name=source_name)
        best, val = orch.run(algo_name)
        print(f"\n✅ 最优解: {best}")
        print(f"✅ 最优值: {val:.6f}")

        # 5. 导出结果
        exporters = registry.list_components('exporter')
        if exporters:
            print("\n??? 导出结果：")
            for i, e in enumerate(exporters):
                print(f"  {i}. {e.name}")
            exp_idx = int(input("选择导出格式 [不导出]: ") or "-1")
            if 0 <= exp_idx < len(exporters):
                exp_name = exporters[exp_idx].name
                exp_cls = registry.get_component(exp_name)
                exporter = exp_cls()

                # 为绘图导出器自动添加 .png 后缀
                algo_short = algo_name.split('.')[-1]
                if 'plot' in exp_name:
                    filepath = f"result_{algo_short}.png"
                else:
                    # 非绘图导出器保留原有逻辑
                    filepath = f"result_{algo_short}.{exp_name.split('.')[-1]}"

                exporter.export(orch.get_history(), filepath)
                print(f"✅ 已保存至 {filepath}")
        # 调试：打印历史记录
        history = orch.get_history()
        print(f"??? 历史记录条数: {len(history)}")
        if len(history) > 0:
            print(f"   第一个点: {history[0].point}, 值: {history[0].value}")
            print(f"   最后一个点: {history[-1].point}, 值: {history[-1].value}")
