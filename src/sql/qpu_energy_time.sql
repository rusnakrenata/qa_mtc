with base as (
select round(qrs.n_filtered_vehicles/10)* 10  as rounded_vehicles,
#c.name, c.center_lon, c.radius_km ,
	qr.energy as qa_energy,
	gr.objective_value as gurobi_energy,
	sr.energy as sa_energy,
	tr.energy as tabu_energy,
	case when cr.status = 'Optimal' then cr.objective_value else NULL end as cbc_energy,
	CONCAT(FORMAT((qr.energy - gr.objective_value)/NULLIF(ABS(gr.objective_value), 0) * 100,2), "%") as delta_energy,
	FORMAT(qr.qubo_density * 100, 2) AS matrix_density
from qubo_run_stats qrs 
inner join iterations i on i.iteration_id = qrs.iteration_id
				and i.run_configs_id = qrs.run_configs_id
inner join run_configs rc on rc.run_configs_id = i.run_configs_id
inner join cities c on c.city_id =rc.city_id
left join  gurobi_results gr on qrs.run_configs_id = gr.run_configs_id
				and qrs.iteration_id = gr.iteration_id
				and qrs.cluster_id = gr.cluster_id
left join qa_results qr on qrs.run_configs_id = qr.run_configs_id
				and qrs.iteration_id = qr.iteration_id
				and qrs.cluster_id = qr.cluster_id
left join sa_results sr on qrs.run_configs_id = sr.run_configs_id
				and qrs.iteration_id = sr.iteration_id
				and qrs.cluster_id = sr.cluster_id
left join tabu_results tr on qrs.run_configs_id = tr.run_configs_id
				and qrs.iteration_id = tr.iteration_id
				and qrs.cluster_id = tr.cluster_id
left join cbc_results cr on qrs.run_configs_id = cr.run_configs_id
				and qrs.iteration_id = cr.iteration_id
				and qrs.cluster_id = cr.cluster_id
where qr.comp_type = 'qpu'
)
select distinct
	rounded_vehicles as n_vehicles,
	count(*) as simulation_count,
	#avg(qa_energy) as qa_energy,
	#avg(gurobi_energy) as gurobi_energy,
	#avg(sa_energy) as sa_energy,
	#avg(tabu_energy) as tabu_energy,
	#avg(cbc_energy) as cbc_energy,
SUM(
    CASE WHEN qa_energy IS NOT NULL
      AND (gurobi_energy IS NULL OR qa_energy <= gurobi_energy)
      AND (sa_energy     IS NULL OR qa_energy <= sa_energy)
      AND (tabu_energy   IS NULL OR qa_energy <= tabu_energy)
      AND (cbc_energy    IS NULL OR qa_energy <= cbc_energy)
    THEN 1 ELSE 0 END
  ) / COUNT(*) * 100.0 AS qa_le_objective_value,
  SUM(
    CASE WHEN gurobi_energy IS NOT NULL
      AND (qa_energy   IS NULL OR gurobi_energy <= qa_energy)
      AND (sa_energy   IS NULL OR gurobi_energy <= sa_energy)
      AND (tabu_energy IS NULL OR gurobi_energy <= tabu_energy)
      AND (cbc_energy  IS NULL OR gurobi_energy <= cbc_energy)
    THEN 1 ELSE 0 END
  ) / COUNT(*) * 100.0 AS gurobi_le_objective_value,
  SUM(
    CASE WHEN sa_energy IS NOT NULL
      AND (qa_energy    IS NULL OR sa_energy <= qa_energy)
      AND (gurobi_energy IS NULL OR sa_energy <= gurobi_energy)
      AND (tabu_energy  IS NULL OR sa_energy <= tabu_energy)
      AND (cbc_energy   IS NULL OR sa_energy <= cbc_energy)
    THEN 1 ELSE 0 END
  ) / COUNT(*) * 100.0 AS sa_le_objective_value,
  SUM(
    CASE WHEN tabu_energy IS NOT NULL
      AND (qa_energy     IS NULL OR tabu_energy <= qa_energy)
      AND (gurobi_energy IS NULL OR tabu_energy <= gurobi_energy)
      AND (sa_energy     IS NULL OR tabu_energy <= sa_energy)
      AND (cbc_energy    IS NULL OR tabu_energy <= cbc_energy)
    THEN 1 ELSE 0 END
  ) / COUNT(*) * 100.0 AS tabu_le_objective_value,
  SUM(
    CASE WHEN cbc_energy IS NOT NULL
      AND (qa_energy     IS NULL OR cbc_energy <= qa_energy)
      AND (gurobi_energy IS NULL OR cbc_energy <= gurobi_energy)      
      AND (sa_energy     IS NULL OR cbc_energy <= sa_energy)
      AND (tabu_energy   IS NULL OR cbc_energy <= tabu_energy)
    THEN 1 ELSE 0 END
  ) / COUNT(*) * 100.0 AS cbc_le_objective_value,
  SUM(
    CASE WHEN qa_energy IS NOT NULL
      AND (cbc_energy     IS NULL OR qa_energy < cbc_energy)
      AND (gurobi_energy IS NULL OR qa_energy < gurobi_energy)      
      AND (sa_energy     IS NULL OR qa_energy < sa_energy)
      AND (tabu_energy   IS NULL OR qa_energy < tabu_energy)
    THEN 1 ELSE 0 END) / COUNT(*) * 100.0 AS qa_best_objective_value,
   SUM(
    CASE WHEN gurobi_energy IS NOT NULL
      AND (qa_energy   IS NULL OR gurobi_energy < qa_energy)
      AND (sa_energy   IS NULL OR gurobi_energy < sa_energy)
      AND (tabu_energy IS NULL OR gurobi_energy < tabu_energy)
      AND (cbc_energy  IS NULL OR gurobi_energy < cbc_energy)
    THEN 1 ELSE 0 END
  ) / COUNT(*) * 100.0 AS gurobi_best_objective_value,
  SUM(
    CASE WHEN sa_energy IS NOT NULL
      AND (qa_energy    IS NULL OR sa_energy < qa_energy)
      AND (gurobi_energy IS NULL OR sa_energy < gurobi_energy)
      AND (tabu_energy  IS NULL OR sa_energy < tabu_energy)
      AND (cbc_energy   IS NULL OR sa_energy < cbc_energy)
    THEN 1 ELSE 0 END
  ) / COUNT(*) * 100.0 AS sa_best_objective_value,
  SUM(
    CASE WHEN tabu_energy IS NOT NULL
      AND (qa_energy     IS NULL OR tabu_energy < qa_energy)
      AND (gurobi_energy IS NULL OR tabu_energy < gurobi_energy)
      AND (sa_energy     IS NULL OR tabu_energy < sa_energy)
      AND (cbc_energy    IS NULL OR tabu_energy < cbc_energy)
    THEN 1 ELSE 0 END
  ) / COUNT(*) * 100.0 AS tabu_best_objective_value,
    SUM(
    CASE WHEN cbc_energy IS NOT NULL
      AND (qa_energy     IS NULL OR cbc_energy < qa_energy)
      AND (gurobi_energy IS NULL OR cbc_energy < gurobi_energy)      
      AND (sa_energy     IS NULL OR cbc_energy < sa_energy)
      AND (tabu_energy   IS NULL OR cbc_energy < tabu_energy)
    THEN 1 ELSE 0 END
  ) / COUNT(*) * 100.0 AS cbc_best_objective_value,
  avg(delta_energy) as delta_energy,
  avg(matrix_density) as matrix_density
from base
group by rounded_vehicles
order by 1;

with base as (
select round(qrs.n_filtered_vehicles/10)* 10  as n_vehicles,
	qr.solver_time as qa_solver_time,
	gr.solver_time as gurobi_solver_time,
	sr.solver_time as sa_solver_time,
	tr.solver_time as tabu_solver_time,
	case when cr.status = 'Optimal' then cr.solver_time else NULL end as cbc_solver_time
from qubo_run_stats qrs 
inner join iterations i on i.iteration_id = qrs.iteration_id
				and i.run_configs_id = qrs.run_configs_id
inner join run_configs rc on rc.run_configs_id = i.run_configs_id
inner join cities c on c.city_id =rc.city_id
left join  gurobi_results gr on qrs.run_configs_id = gr.run_configs_id
				and qrs.iteration_id = gr.iteration_id
				and qrs.cluster_id = gr.cluster_id
left join qa_results qr on qrs.run_configs_id = qr.run_configs_id
				and qrs.iteration_id = qr.iteration_id
				and qrs.cluster_id = qr.cluster_id
left join sa_results sr on qrs.run_configs_id = sr.run_configs_id
				and qrs.iteration_id = sr.iteration_id
				and qrs.cluster_id = sr.cluster_id
left join tabu_results tr on qrs.run_configs_id = tr.run_configs_id
				and qrs.iteration_id = tr.iteration_id
				and qrs.cluster_id = tr.cluster_id
left join cbc_results cr on qrs.run_configs_id = cr.run_configs_id
				and qrs.iteration_id = cr.iteration_id
				and qrs.cluster_id = cr.cluster_id
where qr.comp_type = 'qpu'
)
select distinct
	n_vehicles,
	count(*) as simulation_count,
	#avg(qa_solver_time) as qa_solver_time,
	#avg(gurobi_solver_time) as gurobi_solver_time,
	#avg(sa_solver_time) as sa_solver_time,
	#avg(tabu_solver_time) as tabu_solver_time,
	#avg(cbc_solver_time) as cbc_solver_time,
SUM(
    CASE WHEN qa_solver_time IS NOT NULL
      AND (gurobi_solver_time IS NULL OR qa_solver_time <= gurobi_solver_time)
      AND (sa_solver_time     IS NULL OR qa_solver_time <= sa_solver_time)
      AND (tabu_solver_time   IS NULL OR qa_solver_time <= tabu_solver_time)
      AND (cbc_solver_time    IS NULL OR qa_solver_time <= cbc_solver_time)
    THEN 1 ELSE 0 END
  ) / COUNT(*) * 100.0 AS qa_le_solver_time,
  SUM(
    CASE WHEN gurobi_solver_time IS NOT NULL
      AND (qa_solver_time   IS NULL OR gurobi_solver_time <= qa_solver_time)
      AND (sa_solver_time   IS NULL OR gurobi_solver_time <= sa_solver_time)
      AND (tabu_solver_time IS NULL OR gurobi_solver_time <= tabu_solver_time)
      AND (cbc_solver_time  IS NULL OR gurobi_solver_time <= cbc_solver_time)
    THEN 1 ELSE 0 END
  ) / COUNT(*) * 100.0 AS gurobi_le_solver_time,
  SUM(
    CASE WHEN sa_solver_time IS NOT NULL
      AND (qa_solver_time    IS NULL OR sa_solver_time <= qa_solver_time)
      AND (gurobi_solver_time IS NULL OR sa_solver_time <= gurobi_solver_time)
      AND (tabu_solver_time  IS NULL OR sa_solver_time <= tabu_solver_time)
      AND (cbc_solver_time   IS NULL OR sa_solver_time <= cbc_solver_time)
    THEN 1 ELSE 0 END
  ) / COUNT(*) * 100.0 AS sa_le_solver_time,
  SUM(
    CASE WHEN tabu_solver_time IS NOT NULL
      AND (qa_solver_time     IS NULL OR tabu_solver_time <= qa_solver_time)
      AND (gurobi_solver_time IS NULL OR tabu_solver_time <= gurobi_solver_time)
      AND (sa_solver_time     IS NULL OR tabu_solver_time <= sa_solver_time)
      AND (cbc_solver_time    IS NULL OR tabu_solver_time <= cbc_solver_time)
    THEN 1 ELSE 0 END
  ) / COUNT(*) * 100.0 AS tabu_le_solver_time,
  SUM(
    CASE WHEN cbc_solver_time IS NOT NULL
      AND (qa_solver_time     IS NULL OR cbc_solver_time <= qa_solver_time)
      AND (gurobi_solver_time IS NULL OR cbc_solver_time <= gurobi_solver_time)
      AND (sa_solver_time     IS NULL OR cbc_solver_time <= sa_solver_time)
      AND (tabu_solver_time   IS NULL OR cbc_solver_time <= tabu_solver_time)
    THEN 1 ELSE 0 END
  ) / COUNT(*) * 100.0 AS cbc_le_solver_time
from base
group by n_vehicles
order by 1







