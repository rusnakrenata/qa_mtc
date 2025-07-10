SELECT 
    n_vehicles, 
    lambda_value, 
    assignment_valid, 
    invalid_assignment_vehicles,
    dwave_constraints_check
FROM trafficOptimization.qa_results qr
WHERE run_configs_id = %s 
  AND iteration_id = %s;
SELECT 
    SUM(congestion_all) AS congestion_all, 
    SUM(congestion_post_qa) AS congestion_post_qa, 
    SUM(congestion_shortest_dis) AS congestion_shortest_dist, 
    SUM(congestion_shortest_dur) AS congestion_shortest_dur,
    SUM(congestion_random) AS congestion_random,
    (SUM(congestion_shortest_dur) - SUM(congestion_post_qa)) / SUM(congestion_shortest_dur) * 100 AS pct_qa_shortest_dur_improvement,
    (SUM(congestion_random) - SUM(congestion_post_qa)) / SUM(congestion_random) * 100 AS pct_qa_random__improvement
FROM trafficOptimization.congestion_summary cs
WHERE run_configs_id = %s 
  AND iteration_id = %s;
SELECT 
    SUM(shortest_dur) AS shortest_dur,
    SUM(post_qa_dur) AS post_qa_dur,
    SUM(rnd_dur) AS rnd_dur,
    (SUM(shortest_dur) - SUM(post_qa_dur)) / SUM(shortest_dur) * 100 AS pct_qa_shortest_dur_improvement,
    (SUM(rnd_dur) - SUM(post_qa_dur)) / SUM(rnd_dur) * 100 AS pct_rnd_dur_improvement,
    SUM(shortest_dist) AS shortest_dist,
    SUM(post_qa_dist) AS post_qa_dist,
    SUM(rnd_dist) AS rnd_dist,
    (SUM(shortest_dist) - SUM(post_qa_dist)) / SUM(shortest_dist) * 100 AS pct_qa_shortest_dist_improvement,
    (SUM(rnd_dist) - SUM(post_qa_dist)) / SUM(rnd_dist) * 100 AS pct_rnd_dist_improvement   
FROM trafficOptimization.dist_dur_summary dd
WHERE run_configs_id = %s
  AND iteration_id = %s;
WITH shortest_routes AS (
    SELECT 
        vehicle_id, 
        MIN(duration) AS min_value, 
        run_configs_id, 
        iteration_id
    FROM vehicle_routes
    WHERE run_configs_id = %s 
      AND iteration_id = %s
    GROUP BY vehicle_id, run_configs_id, iteration_id
),
selected_routes AS (
    SELECT 
        vr.vehicle_id, 
        MAX(vr.route_id) AS route_id, 
        ss.route_id AS qa_route_id 
    FROM vehicle_routes vr
    JOIN shortest_routes sr
        ON vr.vehicle_id = sr.vehicle_id 
        AND vr.duration = sr.min_value
    INNER JOIN selected_routes ss 
        ON sr.vehicle_id = ss.vehicle_id
    WHERE vr.run_configs_id = sr.run_configs_id 
      AND vr.iteration_id = sr.iteration_id
    GROUP BY vr.vehicle_id, ss.route_id
)
SELECT 
    vehicle_id, 
    route_id, 
    qa_route_id 
FROM selected_routes
WHERE route_id <> qa_route_id;
