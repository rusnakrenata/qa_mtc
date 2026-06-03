SELECT DISTINCT
ROUND(n_filtered_vehicles / 100) * 100 AS n_vehicles,
COUNT(*) AS n_runs,
AVG(qr.energy) AS hybrid_energy_mean,
STDDEV(qr.energy) AS hybrid_energy_std,
AVG(gr.objective_value) AS gurobi_energy_mean,
STDDEV(gr.objective_value) AS gurobi_energy_std,
AVG(sr.energy) AS sa_energy_mean,
STDDEV(sr.energy) AS sa_energy_std,
AVG(tr.energy) AS tabu_energy_mean,
STDDEV(tr.energy) AS tabu_energy_std,
AVG(CASE WHEN cr.status = 'Optimal' THEN cr.objective_value ELSE NULL END) AS cbc_energy_mean,
STDDEV(CASE WHEN cr.status = 'Optimal' THEN cr.objective_value ELSE NULL END) AS cbc_energy_std,
AVG((qr.energy - gr.objective_value) / NULLIF(ABS(gr.objective_value), 0)) * 100 AS delta_energy_mean,
STDDEV((qr.energy - gr.objective_value) / NULLIF(ABS(gr.objective_value), 0)) * 100 AS delta_energy_std,
STDDEV((qr.energy - gr.objective_value) / NULLIF(ABS(gr.objective_value), 0)) / SQRT(COUNT(*)) * 100 AS delta_energy_se,
AVG((qr.energy - gr.objective_value) / NULLIF(ABS(gr.objective_value), 0)) * 100
- 1.96 * STDDEV((qr.energy - gr.objective_value) / NULLIF(ABS(gr.objective_value), 0)) / SQRT(COUNT(*)) * 100 AS delta_energy_ci95_lower,
AVG((qr.energy - gr.objective_value) / NULLIF(ABS(gr.objective_value), 0)) * 100
+ 1.96 * STDDEV((qr.energy - gr.objective_value) / NULLIF(ABS(gr.objective_value), 0)) / SQRT(COUNT(*)) * 100 AS delta_energy_ci95_upper
FROM qubo_run_stats qrs
INNER JOIN iterations i
ON i.iteration_id = qrs.iteration_id
AND i.run_configs_id = qrs.run_configs_id
INNER JOIN run_configs rc
ON rc.run_configs_id = i.run_configs_id
INNER JOIN cities c
ON c.city_id = rc.city_id
INNER JOIN gurobi_results gr
ON qrs.run_configs_id = gr.run_configs_id
AND qrs.iteration_id = gr.iteration_id
AND qrs.cluster_id = gr.cluster_id
INNER JOIN qa_results qr
ON qrs.run_configs_id = qr.run_configs_id
AND qrs.iteration_id = qr.iteration_id
AND qrs.cluster_id = qr.cluster_id
INNER JOIN sa_results sr
ON qrs.run_configs_id = sr.run_configs_id
AND qrs.iteration_id = sr.iteration_id
AND qrs.cluster_id = sr.cluster_id
INNER JOIN tabu_results tr
ON qrs.run_configs_id = tr.run_configs_id
AND qrs.iteration_id = tr.iteration_id
AND qrs.cluster_id = tr.cluster_id
LEFT JOIN cbc_results cr
ON qrs.run_configs_id = cr.run_configs_id
AND qrs.iteration_id = cr.iteration_id
AND qrs.cluster_id = cr.cluster_id
WHERE qr.comp_type = 'hybrid'
AND n_filtered_vehicles <= 500
GROUP BY ROUND(qrs.n_filtered_vehicles / 100) * 100
ORDER BY 1;