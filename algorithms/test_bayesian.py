import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from algorithms.bayesian import BayesianOptimization


class MockSource:
    def __init__(self, func):
        self.func = func
        self.call_count = 0
        self.query_points = []
    
    def measure(self, point, n_samples=5):
        self.call_count += 1
        self.query_points.append(point.copy())
        return self.func(point)


def test_bayesian_simple_quadratic():
    def quadratic(x):
        return -(x[0] - 0.5)**2 - (x[1] - 0.3)**2
    
    source = MockSource(quadratic)
    algo = BayesianOptimization(source)
    
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    best_point, best_value = algo.optimize(bounds)
    
    assert len(best_point) == 2
    assert best_value <= 0
    assert np.all(best_point >= 0.0)
    assert np.all(best_point <= 1.0)
    
    distance_to_optimum = np.linalg.norm(best_point - np.array([0.5, 0.3]))
    assert distance_to_optimum < 0.3


def test_bayesian_exploration():
    call_locations = []
    
    def tracking_function(x):
        call_locations.append(x.copy())
        return -np.sum(x ** 2)
    
    source = MockSource(tracking_function)
    algo = BayesianOptimization(source)
    
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    best_point, best_value = algo.optimize(bounds)
    
    call_array = np.array(call_locations)
    
    x_range = call_array[:, 0].max() - call_array[:, 0].min()
    y_range = call_array[:, 1].max() - call_array[:, 1].min()
    
    assert x_range > 0.3
    assert y_range > 0.3


def test_bayesian_gaussian_process():
    source = MockSource(lambda x: -np.sum(x ** 2))
    algo = BayesianOptimization(source)
    
    X_train = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_train = np.array([0.0, -2.0])
    
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    mu, sigma = algo._gaussian_process(X_train, y_train, bounds)
    
    test_point = np.array([0.0, 0.0])
    mu_val = mu(test_point)
    sigma_val = sigma(test_point)
    
    assert np.isclose(mu_val, 0.0, atol=0.1)
    assert sigma_val >= 0


if __name__ == '__main__':
    print("Testing Bayesian Optimization Algorithm...")
    
    try:
        test_bayesian_simple_quadratic()
        print("✓ test_bayesian_simple_quadratic passed")
    except AssertionError as e:
        print(f"✗ test_bayesian_simple_quadratic failed: {e}")
    
    try:
        test_bayesian_exploration()
        print("✓ test_bayesian_exploration passed")
    except AssertionError as e:
        print(f"✗ test_bayesian_exploration failed: {e}")
    
    try:
        test_bayesian_gaussian_process()
        print("✓ test_bayesian_gaussian_process passed")
    except AssertionError as e:
        print(f"✗ test_bayesian_gaussian_process failed: {e}")
    
    print("\nAll Bayesian tests completed!")
