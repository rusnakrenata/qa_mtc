import numpy as np
import pandas as pd
from src.modules.qubo_matrix import qubo_matrix

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

def test_qubo_matrix_normalized():
    n, t = 4, 2
    vehicle_ids = list(range(n))
    weights_df = make_weights_df(n, t)
    congestion_df = pd.DataFrame(columns=pd.Index(['edge_id', 'congestion_score']))
    Q, filtered = qubo_matrix(n, t, congestion_df, weights_df, vehicle_ids, lambda_strategy="normalized", fixed_lambda=1.0, filtering_percentage=0.5)
    assert isinstance(Q, dict)
    assert len(filtered) == 2

def test_qubo_matrix_max_weight():
    n, t = 3, 2
    vehicle_ids = list(range(n))
    weights_df = make_weights_df(n, t)
    congestion_df = pd.DataFrame(columns=pd.Index(['edge_id', 'congestion_score']))
    Q, filtered = qubo_matrix(n, t, congestion_df, weights_df, vehicle_ids, lambda_strategy="max_weight", filtering_percentage=1.0)
    assert isinstance(Q, dict)
    assert len(filtered) == 3 