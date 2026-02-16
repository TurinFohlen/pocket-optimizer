import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from orchestrator import Orchestrator, OptimizationConfig
import sources.test_function
import algorithms.pso
import algorithms.simulated_annealing
import algorithms.genetic
import exporters.convergence_plot


def demo_convergence_plot():
    print("="*70)
    print("收敛曲线图修复效果演示")
    print("="*70)
    
    config = OptimizationConfig(
        param_bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        param_names=['x', 'y', 'z'],
        num_samples=3,
        max_evaluations=100
    )
    
    algorithms_to_test = [
        ('algorithm.pso', 'PSO'),
        ('algorithm.simulated_annealing', '模拟退火'),
        ('algorithm.genetic', '遗传算法')
    ]
    
    from registry import registry
    exporter_class = registry.get_component('exporter.convergence_plot')
    exporter = exporter_class()
    
    for algo_name, display_name in algorithms_to_test:
        print(f"\n{'='*70}")
        print(f"测试算法: {display_name}")
        print(f"{'='*70}")
        
        orch = Orchestrator(config, source_name='source.test_function')
        
        print(f"\n运行 {display_name}...")
        best_point, best_value = orch.run(algo_name)
        
        history = orch.get_history()
        
        print(f"\n结果:")
        print(f"  最优点: {best_point}")
        print(f"  最优值: {best_value:.6f}")
        print(f"  评估次数: {len(history)}")
        
        if len(history) > 10:
            filepath = f"convergence_{algo_name.split('.')[-1]}.png"
            exporter.export(history, filepath)
            print(f"\n✅ 收敛曲线图已生成: {filepath}")
        else:
            print(f"\n❌ 评估次数不足,无法生成有效曲线图")
    
    print(f"\n{'='*70}")
    print("演示完成!")
    print(f"{'='*70}")
    print("\n生成的文件:")
    import glob
    for f in glob.glob("convergence_*.png"):
        print(f"  📊 {f}")


if __name__ == '__main__':
    demo_convergence_plot()
