def normalize_congestion_weights(weights_df, n, t, vehicle_ids):
    """
    Converts a DataFrame of congestion weights into a normalized 4D matrix w[i][j][k1][k2].
    Normalization is done over all weights into the [0, 1] range.
    """
    vehicle_ids_index = {vid: idx for idx, vid in enumerate(vehicle_ids)}
    w = [[[[0.0 for _ in range(t)] for _ in range(t)] for _ in range(n)] for _ in range(n)]
    values = []

    for _, row in weights_df.iterrows():
        i = vehicle_ids_index.get(row['vehicle_1'])
        j = vehicle_ids_index.get(row['vehicle_2'])
        k1 = int(row['vehicle_1_route']) - 1
        k2 = int(row['vehicle_2_route']) - 1

        if i is None or j is None or not (0 <= k1 < t) or not (0 <= k2 < t):
            continue

        score = row['weighted_congestion_score']
        values.append(score)
        w[i][j][k1][k2] = score
        w[j][i][k2][k1] = score  # symmetric

    if values:
        min_w = min(values)
        max_w = max(values)
        scale = max_w - min_w + 1e-9
        for i in range(n):
            for j in range(n):
                for k1 in range(t):
                    for k2 in range(t):
                        w[i][j][k1][k2] = (w[i][j][k1][k2] - min_w) / scale
    else:
        min_w = max_w = 0.0

    nonzero_count = sum(
        1 for i in range(n) for j in range(n) for k1 in range(t) for k2 in range(t)
        if w[i][j][k1][k2] > 0
    )

    print(f"|i| = {n}, |j| = {n}, |k1| = {t}, |k2| = {t}")
    print(f"min_w = {min_w:.6f}, max_w = {max_w:.6f}, non-zero weights = {nonzero_count}")

    return w
