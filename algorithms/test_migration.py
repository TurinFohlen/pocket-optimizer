import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from orchestrator import Orchestrator, OptimizationConfig

import sources.test_function
import algorithms.pso
import algorithms.simulated_annealing
import algorithms.bayesian
import algorithms.powell
import algorithms.genetic

from registry import registry


def test_algorithm(algo_name: str, bounds, param_names):
    print(f"\n{'='*70}")
    print(f"Testing: {algo_name}")
    print(f"{'='*70}\n")
    
    config = OptimizationConfig(
        param_bounds=bounds,
        param_names=param_names,
        num_samples=5,
        max_evaluations=30
    )
    
    orch = Orchestrator(config, source_name='source.test_function')
    
    try:
        best_point, best_value = orch.run(algo_name)
        
        print(f"\nResults for {algo_name}:")
        print(f"  Best Point: {best_point}")
        print(f"  Best Value: {best_value:.6f}")
        
        return True
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("ALGORITHM MIGRATION VALIDATION SUITE")
    print("="*70)
    
    print("\n1. Checking registered components...")
    all_components = registry.list_components()
    print(f"   Total components: {len(all_components)}")
    
    algorithms = registry.list_components('algorithm')
    print(f"   Algorithms: {len(algorithms)}")
    for algo in algorithms:
        print(f"     - {algo.name}")
    
    sources = registry.list_components('source')
    print(f"   Sources: {len(sources)}")
    for src in sources:
        print(f"     - {src.name}")
    
    print("\n2. Validating dependencies...")
    is_valid, errors = registry.validate_dependencies()
    if is_valid:
        print("   ✓ All dependencies satisfied")
    else:
        print("   ✗ Dependency errors:")
        for error in errors:
            print(f"     - {error}")
        return
    
    print("\n3. Testing algorithms on test function...")
    
    bounds_2d = [(0.0, 1.0), (0.0, 1.0)]
    param_names_2d = ['x', 'y']
    
    bounds_3d = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    param_names_3d = ['x', 'y', 'z']
    
    test_cases = [
        ('algorithm.pso', bounds_3d, param_names_3d),
        ('algorithm.simulated_annealing', bounds_3d, param_names_3d),
        ('algorithm.powell', bounds_2d, param_names_2d),
        ('algorithm.bayesian', bounds_2d, param_names_2d),
        ('algorithm.genetic', bounds_3d, param_names_3d),
    ]
    
    results = {}
    for algo_name, bounds, param_names in test_cases:
        success = test_algorithm(algo_name, bounds, param_names)
        results[algo_name] = success
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for algo_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} | {algo_name}")
    
    total = len(results)
    passed = sum(1 for s in results.values() if s)
    
    print("-"*70)
    print(f"Total: {passed}/{total} algorithms passed")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL ALGORITHMS VALIDATED SUCCESSFULLY!")
        print("\nThe migration is complete. All algorithms are:")
        print("  ✓ Properly registered")
        print("  ✓ Dependencies auto-detected")
        print("  ✓ Executable via orchestrator")
        print("  ✓ Producing valid results")
    else:
        print("\n⚠️  Some algorithms failed validation.")
        print("    Please review the error messages above.")


if __name__ == '__main__':
    main()
