import sys
#sys.path.insert(0, '/home/claude')

from orchestrator import Orchestrator, OptimizationConfig
from cli_utils import (
    validate_dependencies,
    list_valid_pipelines,
    print_pipelines_table,
    visualize_dependencies
)
from registry import registry

import test_components


def main():
    print("="*70)
    print("OPTIMIZATION SYSTEM - COMPONENT REGISTRY DEMO")
    print("="*70)
    
    print("\n1. Validating dependencies...")
    is_valid, errors = validate_dependencies()
    
    if is_valid:
        print("   ✓ All dependencies satisfied")
    else:
        print("   ✗ Dependency errors:")
        for error in errors:
            print(f"     - {error}")
        return
    
    print("\n2. Discovering available pipelines...")
    pipelines = list_valid_pipelines()
    print_pipelines_table(pipelines)
    
    print("\n3. Visualizing dependency graph...")
    visualize_dependencies()
    
    print("\n4. Running optimization with test pipeline...")
    
    config = OptimizationConfig(
        param_bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        param_names=['x', 'y'],
        num_samples=3,
        max_evaluations=10
    )
    
    orchestrator = Orchestrator(config, source_name='source.test_cli')
    
    print("\n   Running algorithm.test_ga...")
    best_point, best_value = orchestrator.run('algorithm.test_ga')
    
    print(f"\n   Results:")
    print(f"   Best Point: [{best_point[0]:.4f}, {best_point[1]:.4f}]")
    print(f"   Best Value: {best_value:.4f}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
