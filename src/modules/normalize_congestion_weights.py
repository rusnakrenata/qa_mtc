def normalize_congestion_weights(weights_df, n, t, vehicle_ids):
    """
    Converts a DataFrame of congestion weights into a normalized 3D matrix w[i][j][k].
    Normalization is done to scale weights into [0, 1] range.
    """
    vehicle_ids_index = {vid: idx for idx, vid in enumerate(vehicle_ids)}
    w = [[[0.0 for _ in range(t)] for _ in range(n)] for _ in range(n)]
    values = []

    for _, row in weights_df.iterrows():
        i = vehicle_ids_index.get(row['vehicle_1'])
        j = vehicle_ids_index.get(row['vehicle_2'])
        k1 = int(row['vehicle_1_route']) - 1
        k2 = int(row['vehicle_2_route']) - 1

        if i is None or j is None or k1 != k2:
            continue

        k = k1
        score = row['weighted_congestion_score']
        values.append(score)
        w[i][j][k] = score
        w[j][i][k] = score  # symmetric

    if values:
        min_w = min(values)
        max_w = max(values)
        scale = max_w - min_w + 1e-9
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(t):
                    w[i][j][k] = (w[i][j][k] - min_w) / scale
                    w[j][i][k] = w[i][j][k]

    nonzero_count = sum(
    1 for i in range(n) for j in range(n) for k in range(t)
    if w[i][j][k] > 0
    )
    
    print(f"|i| = {n}, |j| = {n}, |k| = {t}")
    print(f"min_w = {min_w:.6f}, max_w = {max_w:.6f}, non-zero weights = {nonzero_count}")

    return w
