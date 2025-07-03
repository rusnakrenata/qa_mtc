select n_vehicles, lambda_value,assignment_valid
from trafficOptimization.qa_results qr 
where run_configs_id = %s and iteration_id = %s;
select sum(congestion_all) as congestion_all, sum(congestion_post_qa) as congestion_post_qa, 
sum(congestion_shortest_dis ) as congestion_shortest_dist, sum(congestion_shortest_dur ) as congestion_shortest_dur,
(sum(congestion_shortest_dur)- sum(congestion_post_qa )) /  sum(congestion_shortest_dur) * 100 as pct_qa_improvement
from trafficOptimization.congestion_summary cs 
where run_configs_id  =%s and iteration_id  =%s;
WITH shortest_routes AS (
    SELECT vehicle_id, min(duration) AS min_value, run_configs_id, iteration_id
    FROM vehicle_routes
    WHERE run_configs_id = %s AND iteration_id = %s
    GROUP BY vehicle_id, run_configs_id, iteration_id
),
selected_routes AS (
    SELECT vr.vehicle_id, max(vr.route_id) as route_id, ss.route_id as qa_route_id 
    FROM vehicle_routes vr
    JOIN shortest_routes sr
    inner join selected_routes ss on sr.vehicle_id = ss.vehicle_id
    ON vr.vehicle_id = sr.vehicle_id AND vr.duration = sr.min_value
    WHERE vr.run_configs_id = sr.run_configs_id AND vr.iteration_id = sr.iteration_id
    GROUP BY vr.vehicle_id, ss.route_id
)
SELECT vehicle_id, route_id, qa_route_id FROM selected_routes
where route_id <> qa_route_id;