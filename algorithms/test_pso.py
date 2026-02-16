import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from algorithms.pso import PSOAlgorithm


class MockSource:
    def __init__(self, func):
        self.func = func
        self.call_count = 0
    
    def measure(self, point, n_samples=5):
        self.call_count += 1
        return self.func(point)


def test_pso_sphere_function():
    def sphere(x):
        return -np.sum(x ** 2)
    
    source = MockSource(sphere)
    algo = PSOAlgorithm(source)
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    best_point, best_value = algo.optimize(bounds)
    
    assert len(best_point) == 2
    assert best_value <= 0
    assert np.all(best_point >= -5.0)
    assert np.all(best_point <= 5.0)
    assert source.call_count > 0
    
    distance_to_optimum = np.linalg.norm(best_point)
    assert distance_to_optimum < 1.0


def test_pso_bounds_respected():
    def constant(x):
        return 1.0
    
    source = MockSource(constant)
    algo = PSOAlgorithm(source)
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    best_point, best_value = algo.optimize(bounds)
    
    assert np.all(best_point >= 0.0)
    assert np.all(best_point <= 1.0)


def test_pso_deterministic_with_seed():
    def quadratic(x):
        return -(x[0] - 0.5)**2 - (x[1] - 0.3)**2
    
    source1 = MockSource(quadratic)
    source2 = MockSource(quadratic)
    
    np.random.seed(42)
    algo1 = PSOAlgorithm(source1)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    result1 = algo1.optimize(bounds)
    
    np.random.seed(42)
    algo2 = PSOAlgorithm(source2)
    result2 = algo2.optimize(bounds)
    
    assert np.allclose(result1[0], result2[0])
    assert np.isclose(result1[1], result2[1])


if __name__ == '__main__':
    print("Testing PSO Algorithm...")
    
    try:
        test_pso_sphere_function()
        print("✓ test_pso_sphere_function passed")
    except AssertionError as e:
        print(f"✗ test_pso_sphere_function failed: {e}")
    
    try:
        test_pso_bounds_respected()
        print("✓ test_pso_bounds_respected passed")
    except AssertionError as e:
        print(f"✗ test_pso_bounds_respected failed: {e}")
    
    try:
        test_pso_deterministic_with_seed()
        print("✓ test_pso_deterministic_with_seed passed")
    except AssertionError as e:
        print(f"✗ test_pso_deterministic_with_seed failed: {e}")
    
    print("\nAll PSO tests completed!")
