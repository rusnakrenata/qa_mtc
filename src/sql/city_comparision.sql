with base as (select 	
	round(qrs.n_filtered_vehicles/500)* 500 as rounded_vehicles,
	qr.n_vehicles,
	radius_km,
	qr.energy,
	qr.duration as qa_model_duration,
    qr.solver_time as qa_solver_time,
	gr.objective_value,
	gr.duration as gurobi_model_duration,
    gr.solver_time as gr_solver_time,
    gr.time_limit_seconds,
	gr.best_bound,
	gr.gap,
	c.name,
	FORMAT(qr.qubo_density * 100, 2) AS matrix_density,
    CONCAT(FORMAT((qr.energy - gr.objective_value)/NULLIF(ABS(gr.objective_value), 0) * 100,2), "%") as delta_energy
from qa_results qr 
inner join  gurobi_results gr on qr.run_configs_id = gr.run_configs_id
				and qr.iteration_id = gr.iteration_id
				and qr.cluster_id = gr.cluster_id
inner join qubo_run_stats qrs on qrs.run_configs_id = qr.run_configs_id
				and qrs.iteration_id = qr.iteration_id
				and qrs.cluster_id = qr.cluster_id
inner join iterations i on i.iteration_id = qr.iteration_id
				and i.run_configs_id = qr.run_configs_id
inner join run_configs rc on rc.run_configs_id = i.run_configs_id
inner join cities c on c.city_id =rc.city_id
where qr.assignment_valid = 1 or n_filtered_vehicles  > 4500
and comp_type = 'hybrid'
and qrs.n_vehicles <> n_filtered_vehicles
)
select 
	rounded_vehicles as veh,
	name,
	#n_vehicles,
	#radius_km ,
	#count(rounded_vehicles) as cnt,
	#avg(matrix_density) as avg_density,
	avg(delta_energy) as avg_delta_energy
	#sum(case when delta_energy <= 0 then 1 else 0 end) as n_qa_better,
	#avg(case when delta_energy <= 0 then matrix_density else NULL end) as qa_better_density,
	#avg(case when delta_energy > 0 then delta_energy else NULL end) as delta_energy_if_worse,
	#avg(case when delta_energy > 0 then matrix_density else NULL end) as qa_worse_density,
	#sum(case when delta_energy <=0  then 1 else 0 end) as delta_energy_u_50,
	#avg(case when delta_energy <=0  then matrix_density else NULL end) as u_50_density,
	#case when avg(case when delta_energy <=0.5  then matrix_density else NULL end) > avg(case when delta_energy > 0.5 then matrix_density else NULL end) then 1 else 0 end as trend
from base
where (name like '%Ko%' or name like '%Car%') and rounded_vehicles <= 9000
#where rounded_vehicles 	in (400,700,1000,1100)
group by rounded_vehicles, name 
order by 1

