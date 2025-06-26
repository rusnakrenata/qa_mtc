import numpy as np
import pandas as pd
from src.modules.congestion_weights import congestion_weights
from src.modules.normalize_congestion_weights import normalize_congestion_weights

def make_weights_df(n, t):
    data = []
    for i in range(n):
        for j in range(n):
            for k1 in range(1, t+1):
                for k2 in range(1, t+1):
                    data.append({
                        'vehicle1': i,
                        'vehicle2': j,
                        'vehicle1_route': k1,
                        'vehicle2_route': k2,
                        'weighted_congestion_score': float(i + j + k1 + k2)
                    })
    return pd.DataFrame(data)

def test_congestion_weights_symmetry():
    n, t = 3, 2
    vehicle_ids = list(range(n))
    weights_df = make_weights_df(n, t)
    w, max_w = congestion_weights(weights_df, n, t, vehicle_ids)
    for i in range(n):
        for j in range(n):
            for k1 in range(t):
                for k2 in range(t):
                    assert w[i][j][k1][k2] == w[j][i][k2][k1]

def test_normalize_congestion_weights_range():
    n, t = 3, 2
    vehicle_ids = list(range(n))
    weights_df = make_weights_df(n, t)
    w = normalize_congestion_weights(weights_df, n, t, vehicle_ids)
    arr = np.array(w)
    assert arr.min() >= 0.0 and arr.max() <= 1.0
    assert arr.shape == (n, n, t, t) 