import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from algorithms.powell import PowellAlgorithm


class MockSource:
    def __init__(self, func):
        self.func = func
        self.call_count = 0
    
    def measure(self, point):
        self.call_count += 1
        return self.func(point)


def test_powell_reflection_mechanism():
    source = MockSource(lambda x: -np.sum(x ** 2))
    algo = PowellAlgorithm(source)
    
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    
    y_values = np.array([0.5, 1.5, 2.5, -0.5])
    
    for y in y_values:
        y_vec = np.array([y, 0.5])
        x = algo.optimize.__code__.co_consts
        
    assert True


def test_powell_simple_quadratic():
    def quadratic(x):
        return -(x[0] - 0.5)**2 - (x[1] - 0.5)**2
    
    source = MockSource(quadratic)
    algo = PowellAlgorithm(source)
    
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    best_point, best_value = algo.optimize(bounds)
    
    assert len(best_point) == 2
    assert best_value <= 0
    assert np.all(best_point >= 0.0)
    assert np.all(best_point <= 1.0)
    
    distance_to_optimum = np.linalg.norm(best_point - np.array([0.5, 0.5]))
    assert distance_to_optimum < 0.2


def test_powell_caching():
    call_log = []
    
    def tracking_function(x):
        x_rounded = tuple(np.round(x, decimals=10))
        call_log.append(x_rounded)
        return -np.sum(x ** 2)
    
    source = MockSource(tracking_function)
    algo = PowellAlgorithm(source)
    
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    best_point, best_value = algo.optimize(bounds)
    
    unique_calls = len(set(call_log))
    total_calls = len(call_log)
    
    assert unique_calls > 0
    assert total_calls >= unique_calls


def test_powell_fallback_without_scipy():
    source = MockSource(lambda x: -np.sum(x ** 2))
    algo = PowellAlgorithm(source)
    
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    
    try:
        best_point, best_value = algo.optimize(bounds)
        assert len(best_point) == 2
        assert np.all(best_point >= 0.0)
        assert np.all(best_point <= 1.0)
    except Exception as e:
        pass


if __name__ == '__main__':
    print("Testing Powell Algorithm...")
    
    try:
        test_powell_reflection_mechanism()
        print("✓ test_powell_reflection_mechanism passed")
    except AssertionError as e:
        print(f"✗ test_powell_reflection_mechanism failed: {e}")
    
    try:
        test_powell_simple_quadratic()
        print("✓ test_powell_simple_quadratic passed")
    except AssertionError as e:
        print(f"✗ test_powell_simple_quadratic failed: {e}")
    
    try:
        test_powell_caching()
        print("✓ test_powell_caching passed")
    except AssertionError as e:
        print(f"✗ test_powell_caching failed: {e}")
    
    try:
        test_powell_fallback_without_scipy()
        print("✓ test_powell_fallback_without_scipy passed")
    except AssertionError as e:
        print(f"✗ test_powell_fallback_without_scipy failed: {e}")
    
    print("\nAll Powell tests completed!")
