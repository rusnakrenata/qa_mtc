import numpy as np
import pytest
from src.modules.filter_routes_for_qubo import filter_routes_for_qubo

def make_toy_weights(n, t, seed=42):
    np.random.seed(seed)
    return np.random.rand(n, n, t, t)

def test_top_score_selection():
    n, t = 10, 3
    vehicle_ids = list(range(n))
    w = make_toy_weights(n, t)
    selected = filter_routes_for_qubo(t, vehicle_ids, w, filtering_percentage=0.2, method="top_score")
    assert len(selected) == 2
    assert all(i in vehicle_ids for i in selected)

def test_greedy_selection():
    n, t = 8, 2
    vehicle_ids = list(range(n))
    w = make_toy_weights(n, t)
    selected = filter_routes_for_qubo(t, vehicle_ids, w, filtering_percentage=0.25, method="greedy")
    assert len(selected) == 2
    assert all(i in vehicle_ids for i in selected)

def test_zero_filtering_raises():
    n, t = 5, 2
    vehicle_ids = list(range(n))
    w = make_toy_weights(n, t)
    with pytest.raises(ValueError):
        filter_routes_for_qubo(t, vehicle_ids, w, filtering_percentage=0.0)

def test_max_qubo_size():
    n, t = 10, 2
    vehicle_ids = list(range(n))
    w = make_toy_weights(n, t)
    selected = filter_routes_for_qubo(t, vehicle_ids, w, filtering_percentage=1.0, max_qubo_size=8)
    assert len(selected) == 4  # 8 // 2 = 4 