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
    --SUM(congestion_all) AS congestion_all, 
    SUM(congestion_post_qa) AS congestion_post_qa, 
    --SUM(congestion_shortest_dis) AS congestion_shortest_dist, 
    SUM(congestion_shortest_dur) AS congestion_shortest_dur,
    SUM(congestion_random) AS congestion_random,
    SUM(congestion_post_gurobi) as congestion_post_gurobi,
    (SUM(congestion_shortest_dur) - SUM(congestion_post_qa)) / SUM(congestion_shortest_dur) * 100 AS pct_qa_shortest_dur_improvement,
    (SUM(congestion_random) - SUM(congestion_post_qa)) / SUM(congestion_random) * 100 AS pct_qa_random__improvement
FROM trafficOptimization.congestion_summary cs
WHERE run_configs_id = %s 
  AND iteration_id = %s;
WITH cong as (
SELECT run_configs_id,
iteration_id,
SUM(congestion_post_qa ) as cong_post_qa,
SUM(congestion_post_gurobi) as cong_post_gurobi,
SUM(congestion_random) as cong_random,
SUM(congestion_shortest_dur) as cong_shortest_dur
FROM trafficOptimization.congestion_summary cs 
WHERE run_configs_id = %s
  AND iteration_id = %s
)
SELECT 
	SUM(post_qa_dur) + SUM(cong_post_qa)*10*0.5 as post_qa_DUR_ADJ, 
	SUM(post_gurobi_dur) + SUM(cong_post_gurobi)*10*0.5 as post_gurobi_DUR_ADJ, 
    SUM(shortest_dur) AS shortest_dur,
    SUM(post_qa_dur) AS post_qa_dur,
    SUM(post_gurobi_dur) as post_gurobi_dur,
    SUM(rnd_dur) AS rnd_dur,
    SUM(cong_shortest_dur) as cong_shortest_dur,
    SUM(cong_post_qa) as cong_post_qa,
    SUM(cong_post_gurobi) as cong_post_gurobi,
    SUM(cong_random) as cong_random
FROM trafficOptimization.dist_dur_summary dd
JOIN cong cs ON  dd.run_configs_id = cs.run_configs_id
  AND dd.iteration_id= cs.iteration_id;
WITH 
selected_routes AS (
    SELECT 
        vr.vehicle_id, 
        ss.route_id AS qa_route_id,
        gr.route_id as gurobi_route_id
    FROM vehicle_routes vr
    INNER JOIN selected_routes ss 
        ON ss.vehicle_id = vr.vehicle_id
    INNER JOIN gurobi_routes gr 
        ON gr.vehicle_id = vr.vehicle_id
    WHERE vr.run_configs_id = ss.run_configs_id 
      AND vr.iteration_id = ss.iteration_id
      AND vr.run_configs_id = gr.run_configs_id 
      AND vr.iteration_id = gr.iteration_id
      AND vr.run_configs_id = %s
      AND vr.iteration_id = %s
    GROUP BY vr.vehicle_id
)
SELECT 
   vehicle_id, qa_route_id, gurobi_route_id
FROM selected_routes
WHERE gurobi_route_id <> qa_route_id;
 