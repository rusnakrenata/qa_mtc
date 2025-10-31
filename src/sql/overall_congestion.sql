WITH vars AS (
    SELECT distinct run_configs_id, iteration_id
    from qa_results
    where comp_type = 'hybrid'
), 
rc AS (
	SELECT rc.run_configs_id, iteration_id, city_id, n_vehicles
	FROM run_configs rc
	INNER JOIN iterations i on i.run_configs_id = rc.run_configs_id
),
city as (
	SELECT *
	FROM cities
),
routes_with_min AS (
    SELECT 
        vr.run_configs_id,
        vr.iteration_id,
        vr.vehicle_id,
        MIN(vr.duration) AS min_duration
    FROM vehicle_routes vr
    JOIN vars ON vr.run_configs_id = vars.run_configs_id
             AND vr.iteration_id = vars.iteration_id
    GROUP BY vr.run_configs_id, vr.iteration_id, vr.vehicle_id
),
penalties AS (
    SELECT
        a.run_configs_id,
        a.iteration_id,
        a.vehicle_id,
        a.route_id,
        a.duration - b.min_duration AS penalty
    FROM vehicle_routes a
    JOIN routes_with_min b 
        ON a.vehicle_id = b.vehicle_id
       AND a.run_configs_id = b.run_configs_id
       AND a.iteration_id = b.iteration_id
),
qa AS (
    SELECT SUM(p.penalty) AS qa_penalty
    	,p.run_configs_id, p.iteration_id
    FROM qa_selected_routes sr
    JOIN penalties p 
        ON p.vehicle_id = sr.vehicle_id
       AND p.route_id = sr.route_id
       AND p.run_configs_id = sr.run_configs_id
       AND p.iteration_id = sr.iteration_id
    GROUP BY p.run_configs_id, p.iteration_id
),
sa AS (
    SELECT SUM(p.penalty) AS sa_penalty
    ,p.run_configs_id, p.iteration_id
    FROM sa_selected_routes sr
    JOIN penalties p 
        ON p.vehicle_id = sr.vehicle_id
       AND p.route_id = sr.route_id
       AND p.run_configs_id = sr.run_configs_id
       AND p.iteration_id = sr.iteration_id
     GROUP BY p.run_configs_id, p.iteration_id
),
tabu AS (
    SELECT SUM(p.penalty) AS tabu_penalty
    ,p.run_configs_id, p.iteration_id
    FROM tabu_selected_routes sr
    JOIN penalties p 
        ON p.vehicle_id = sr.vehicle_id
       AND p.route_id = sr.route_id
       AND p.run_configs_id = sr.run_configs_id
       AND p.iteration_id = sr.iteration_id
     GROUP BY p.run_configs_id, p.iteration_id
),
gurobi AS (
    SELECT SUM(p.penalty) AS gurobi_penalty
    ,p.run_configs_id, p.iteration_id
    FROM gurobi_routes sr
    JOIN penalties p 
        ON p.vehicle_id = sr.vehicle_id
       AND p.route_id = sr.route_id
       AND p.run_configs_id = sr.run_configs_id
       AND p.iteration_id = sr.iteration_id
     GROUP BY p.run_configs_id, p.iteration_id
),
cbc AS (
    SELECT SUM(p.penalty) AS cbc_penalty
    ,p.run_configs_id, p.iteration_id
    FROM cbc_routes sr
    JOIN penalties p 
        ON p.vehicle_id = sr.vehicle_id
       AND p.route_id = sr.route_id
       AND p.run_configs_id = sr.run_configs_id
       AND p.iteration_id = sr.iteration_id
     GROUP BY p.run_configs_id, p.iteration_id
),
random AS (
    SELECT SUM(p.penalty) AS random_penalty
    ,p.run_configs_id, p.iteration_id
    FROM random_routes sr
    JOIN penalties p 
        ON p.vehicle_id = sr.vehicle_id
       AND p.route_id = sr.route_id
       AND p.run_configs_id = sr.run_configs_id
       AND p.iteration_id = sr.iteration_id
     GROUP BY p.run_configs_id, p.iteration_id
),
cong AS (
    SELECT 
    	vars.run_configs_id, vars.iteration_id,
        SUM(cs.congestion_post_qa) AS cong_post_qa,
        SUM(cs.congestion_post_sa) AS cong_post_sa,
        SUM(cs.congestion_post_tabu) AS cong_post_tabu,
        SUM(cs.congestion_post_gurobi) AS cong_post_gurobi,
        SUM(cs.congestion_post_cbc) AS cong_post_cbc,
        SUM(cs.congestion_random) AS cong_random,
        SUM(cs.congestion_shortest_dur) AS cong_shortest_dur
    FROM trafficOptimization.congestion_summary cs
    JOIN vars 
        ON cs.run_configs_id = vars.run_configs_id
       AND cs.iteration_id = vars.iteration_id
    GROUP BY vars.run_configs_id, vars.iteration_id
)
SELECT #qa.run_configs_id, qa.iteration_id, city.name, 
	rc.n_vehicles,
    avg(qa.qa_penalty + cong.cong_post_qa) AS QA_HS_COST,
    avg(gurobi.gurobi_penalty + cong.cong_post_gurobi) AS GUROBI_COST,
    avg(random.random_penalty + cong.cong_random) AS RANDOM_COST,
    avg(0 + cong.cong_shortest_dur) AS SHORTEST_DUR_COST,
    CONCAT(FORMAT(AVG((qa.qa_penalty + cong.cong_post_qa - gurobi.gurobi_penalty - cong.cong_post_gurobi)/NULLIF(ABS(gurobi.gurobi_penalty + cong.cong_post_gurobi), 0)) * 100,2), "%") as delta_cost
	#cong.*
FROM qa
JOIN gurobi ON qa.run_configs_id = gurobi.run_configs_id and qa.iteration_id = gurobi.iteration_id
JOIN random ON qa.run_configs_id = random.run_configs_id and qa.iteration_id = random.iteration_id
JOIN cong ON qa.run_configs_id = cong.run_configs_id and qa.iteration_id = cong.iteration_id
JOIN vars ON qa.run_configs_id = vars.run_configs_id and qa.iteration_id = vars.iteration_id
JOIN rc on qa.run_configs_id = rc.run_configs_id and qa.iteration_id = rc.iteration_id
JOIN city ON rc.city_id = city.city_id
group by n_vehicles
order by 1


