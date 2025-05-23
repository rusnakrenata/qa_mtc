SET @dist_thresh := 10.0;          -- km
SET @speed_diff_thresh := 5.0;     -- km/h
SET @iteration := 1;
SET @run_configs_id := 5;

-- Step 1: All available vehicle-route combinations
WITH vehicle_routes_all AS (
    SELECT vehicle_id, route_id, distance, duration
    FROM trafficOptimization.vehicle_routes
    WHERE iteration_id = @iteration
    AND run_configs_id = @run_configs_id
),-- Step 2: Identify minimal (optimal) route duration and distance per vehicle and route
-- so we will know which route is optimal with respect to duration or distance
optimal_routes_wr_duration AS (
    SELECT a.vehicle_id, b.route_id, a.min_duration
    FROM (
		SELECT vehicle_id,
	           MIN(duration) AS min_duration
	    FROM vehicle_routes_all
	    GROUP BY vehicle_id) a
	 INNER JOIN vehicle_routes_all b on a.vehicle_id = b.vehicle_id
	 								AND a.min_duration = b.duration
),
optimal_routes_wr_distance AS (
    SELECT a.vehicle_id, b.route_id, a.min_distance
    FROM (
		SELECT vehicle_id,
	           MIN(distance) AS min_distance
	    FROM vehicle_routes_all
	    GROUP BY vehicle_id) a
	 INNER JOIN vehicle_routes_all b on a.vehicle_id = b.vehicle_id
	 								AND a.min_distance = b.distance
),-- Step 3: Combine all route pairs with optimal baselines
all_route_combinations AS (
    SELECT 
        a.vehicle_id AS vehicle_id_1,
        a.route_id AS route_id_1,
        a.distance AS distance_1,
        a.duration AS duration_1,
        opt1_dis.min_distance AS optimal_distance_1,
        opt1_dis.route_id AS optimal_route_wr_dis_1,
        opt1_dur.min_duration AS optimal_duration_1,
        opt1_dur.route_id AS optimal_route_wr_dur_1,
        b.vehicle_id AS vehicle_id_2,
        b.route_id AS route_id_2,
        b.distance AS distance_2,
        b.duration AS duration_2,
        opt2_dis.min_distance AS optimal_distance_2,
        opt2_dis.route_id AS optimal_route_wr_dis_2,
        opt2_dur.min_duration AS optimal_duration_2,
        opt2_dur.route_id AS optimal_route_wr_dur_2,
        (a.distance + b.distance) AS total_distance,
        (a.duration + b.duration) AS total_duration,
        (opt1_dis.min_distance + opt2_dis.min_distance) AS optimal_total_distance,
        (opt1_dur.min_duration + opt2_dur.min_duration) AS optimal_total_duration
    FROM vehicle_routes_all a
    JOIN vehicle_routes_all b ON a.vehicle_id < b.vehicle_id
    JOIN optimal_routes_wr_duration opt1_dur ON a.vehicle_id = opt1_dur.vehicle_id
	JOIN optimal_routes_wr_distance opt1_dis ON a.vehicle_id = opt1_dis.vehicle_id
    JOIN optimal_routes_wr_duration opt2_dur ON b.vehicle_id = opt2_dur.vehicle_id
    JOIN optimal_routes_wr_distance opt2_dis ON b.vehicle_id = opt2_dis.vehicle_id
),-- Step 4: Pairwise route points comparison
pairwise AS (
    SELECT
        a.time,
        a.edge_id,
        a.cardinal,
        a.vehicle_id AS vehicle_id_1,
        a.route_id AS route_id_1,
        b.vehicle_id AS vehicle_id_2,
        b.route_id AS route_id_2,
        a.speed AS speed_1,
        b.speed AS speed_2,
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
)-- Final Step: Compute congestion metrics
SELECT *
FROM (
SELECT
    arc.vehicle_id_1,
    arc.route_id_1,
    arc.vehicle_id_2,
    arc.route_id_2,
    COUNT(CASE WHEN pw.distance < @dist_thresh AND pw.speed_diff < @speed_diff_thresh THEN 1 END) AS congestion_events,
    COALESCE(SUM(CASE 
        WHEN pw.distance < @dist_thresh AND pw.speed_diff < @speed_diff_thresh THEN 
            (1 / ((1 + pw.distance) * (1 + pw.speed_diff))) -- see bib/Weighted congestion
        ELSE 0
    END),0) AS weighted_congestion_score,
    -- Realistic deltas using minimal (optimal) baselines
    (arc.total_duration - arc.optimal_total_duration) AS delta_duration,
    (arc.total_distance - arc.optimal_total_distance) AS delta_distance
FROM all_route_combinations arc
LEFT JOIN pairwise pw
    ON arc.vehicle_id_1 = pw.vehicle_id_1
    AND arc.route_id_1 = pw.route_id_1
    AND arc.vehicle_id_2 = pw.vehicle_id_2
    AND arc.route_id_2 = pw.route_id_2
GROUP BY
    arc.vehicle_id_1,
    arc.route_id_1,
    arc.vehicle_id_2,
    arc.route_id_2 ) a

/*
-- Step 1: All available vehicle-route combinations
WITH vehicle_routes_all AS (
    SELECT vehicle_id, route_id, distance, duration
    FROM trafficOptimization.vehicle_routes
    WHERE iteration_id = 11
),-- Step 2: Identify minimal (optimal) route duration and distance per vehicle
optimal_routes AS (
    SELECT vehicle_id,
           MIN(duration) AS min_duration,
           MIN(distance) AS min_distance
    FROM vehicle_routes_all
    GROUP BY vehicle_id
),-- Step 3: Combine all route pairs with optimal baselines
all_route_combinations AS (
    SELECT 
        a.vehicle_id AS vehicle_id_1,
        a.route_id AS route_id_1,
        a.distance AS distance_1,
        a.duration AS duration_1,
        opt1.min_distance AS optimal_distance_1,
        opt1.min_duration AS optimal_duration_1,
        b.vehicle_id AS vehicle_id_2,
        b.route_id AS route_id_2,
        b.distance AS distance_2,
        b.duration AS duration_2,
        opt2.min_distance AS optimal_distance_2,
        opt2.min_duration AS optimal_duration_2,
        (a.distance + b.distance) AS total_distance,
        (a.duration + b.duration) AS total_duration,
        (opt1.min_distance + opt2.min_distance) AS optimal_total_distance,
        (opt1.min_duration + opt2.min_duration) AS optimal_total_duration
    FROM vehicle_routes_all a
    JOIN optimal_routes opt1 ON a.vehicle_id = opt1.vehicle_id
    JOIN vehicle_routes_all b ON a.vehicle_id < b.vehicle_id
    JOIN optimal_routes opt2 ON b.vehicle_id = opt2.vehicle_id
),-- Step 4: Pairwise route points comparison
pairwise AS (
    SELECT
        a.time,
        a.edge_id,
        a.cardinal,
        a.vehicle_id AS vehicle_id_1,
        a.route_id AS route_id_1,
        b.vehicle_id AS vehicle_id_2,
        b.route_id AS route_id_2,
        a.speed AS speed_1,
        b.speed AS speed_2,
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
    WHERE a.iteration_id = 11
      AND b.iteration_id = 11
)-- Final Step: Compute congestion metrics
SELECT
    arc.vehicle_id_1,
    arc.route_id_1,
    arc.vehicle_id_2,
    arc.route_id_2,
    COUNT(CASE WHEN pw.distance < 10 AND pw.speed_diff < 5 THEN 1 END) AS congestion_events,
    COALESCE(SUM(CASE 
        WHEN pw.distance < 10 AND pw.speed_diff < 5 THEN 
            (1 / ((1 + pw.distance) * (1 + pw.speed_diff))) -- see bib/Weighted congestion
        ELSE 0
    END),0) AS weighted_congestion_score,
    -- Realistic deltas using minimal (optimal) baselines
    (arc.total_duration - arc.optimal_total_duration) AS delta_duration,
    (arc.total_distance - arc.optimal_total_distance) AS delta_distance
FROM all_route_combinations arc
LEFT JOIN pairwise pw
    ON arc.vehicle_id_1 = pw.vehicle_id_1
    AND arc.route_id_1 = pw.route_id_1
    AND arc.vehicle_id_2 = pw.vehicle_id_2
    AND arc.route_id_2 = pw.route_id_2
GROUP BY
    arc.vehicle_id_1,
    arc.route_id_1,
    arc.vehicle_id_2,
    arc.route_id_2
  ORDER BY 5 desc;
*/
