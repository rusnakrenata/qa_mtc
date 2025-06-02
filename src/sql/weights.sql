SET @dist_thresh := 10.0;          -- km
SET @speed_diff_thresh := 5.0;     -- km/h
SET @iteration := 1;
SET @run_configs_id := 12; 
 
SELECT
	vehicle_1,
	vehicle_2,
	vehicle_1_route,
	vehicle_2_route,
    --edge_id,
    SUM(CASE 
        WHEN distance < @dist_thresh AND speed_diff < @speed_diff_thresh THEN 
            (1 / ((1 + distance) * (1 + speed_diff)))
        ELSE 0
    END) AS weighted_congestion_score
FROM (
    SELECT
        a.edge_id,
        a.vehicle_id as vehicle_1,
        b.vehicle_id as vehicle_2,
        a.route_id as vehicle_1_route,
        b.route_id as vehicle_2_route,
        6371 * 2 * ASIN(SQRT(
            POW(SIN(RADIANS(b.lat - a.lat) / 2), 2) +
            COS(RADIANS(a.lat)) * COS(RADIANS(b.lat)) *
            POW(SIN(RADIANS(b.lon - a.lon) / 2), 2)
        )) AS distance,
        ABS(a.speed - b.speed) AS speed_diff
    FROM trafficOptimization.route_points a
    JOIN trafficOptimization.route_points b
        ON a.time = b.time
        AND a.edge_id = b.edge_id
        AND a.cardinal = b.cardinal
        AND a.vehicle_id < b.vehicle_id
    WHERE a.iteration_id = @iteration
    AND b.iteration_id = @iteration
    AND a.run_configs_id = @run_configs_id
    AND b.run_configs_id = @run_configs_id
) AS pairwise
GROUP BY vehicle_1,
	vehicle_2, 
	vehicle_1_route,
	vehicle_2_route,
	--edge_id
    ;