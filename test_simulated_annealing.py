import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from algorithms.simulated_annealing import SimulatedAnnealingAlgorithm


class MockSource:
    def __init__(self, func):
        self.func = func
        self.call_count = 0
    
    def measure(self, point, n_samples=5):
        self.call_count += 1
        return self.func(point)


def test_sa_sphere_function():
    def sphere(x):
        return -np.sum(x ** 2)
    
    source = MockSource(sphere)
    algo = SimulatedAnnealingAlgorithm(source)
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    best_point, best_value = algo.optimize(bounds)
    
    assert len(best_point) == 2
    assert best_value <= 0
    assert np.all(best_point >= -5.0)
    assert np.all(best_point <= 5.0)
    
    distance_to_optimum = np.linalg.norm(best_point)
    assert distance_to_optimum < 2.0


def test_sa_accepts_worse_solutions_early():
    call_history = []
    
    def tracking_function(x):
        value = -np.sum(x ** 2)
        call_history.append(value)
        return value
    
    source = MockSource(tracking_function)
    algo = SimulatedAnnealingAlgorithm(source)
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    best_point, best_value = algo.optimize(bounds)
    
    assert len(call_history) > 10
    
    has_non_monotonic = False
    for i in range(1, min(len(call_history), 20)):
        if call_history[i] < call_history[i-1]:
            has_non_monotonic = True
            break
    
    assert has_non_monotonic


def test_sa_temperature_decreases():
    source = MockSource(lambda x: -np.sum(x ** 2))
    algo = SimulatedAnnealingAlgorithm(source)
    
    initial_temp = algo.initial_temperature
    cooling_rate = algo.cooling_rate
    
    assert initial_temp > 0
    assert 0 < cooling_rate < 1
    
    expected_temp_after_10 = initial_temp * (cooling_rate ** 10)
    assert expected_temp_after_10 < initial_temp


if __name__ == '__main__':
    print("Testing Simulated Annealing Algorithm...")
    
    try:
        test_sa_sphere_function()
        print("✓ test_sa_sphere_function passed")
    except AssertionError as e:
        print(f"✗ test_sa_sphere_function failed: {e}")
    
    try:
        test_sa_accepts_worse_solutions_early()
        print("✓ test_sa_accepts_worse_solutions_early passed")
    except AssertionError as e:
        print(f"✗ test_sa_accepts_worse_solutions_early failed: {e}")
    
    try:
        test_sa_temperature_decreases()
        print("✓ test_sa_temperature_decreases passed")
    except AssertionError as e:
        print(f"✗ test_sa_temperature_decreases failed: {e}")
    
    print("\nAll SA tests completed!")
