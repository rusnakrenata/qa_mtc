select distinct
	round(n_filtered_vehicles/10)* 10 as n_vehicles,
	count(*),
	avg(qr.energy) as 'qa qpu_energy',
	avg(gr.objective_value) as gurobi_energy,
	avg(sr.energy) as sa_energy,
	avg(tr.energy) as tabu_energy,
	avg(case when cr.status = 'Optimal' then cr.objective_value else NULL end) as cbc_energy,
	FORMAT(AVG((qr.energy - gr.objective_value)/NULLIF(ABS(gr.objective_value), 0)) * 100,2) as delta_energy
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
group by round(qrs.n_filtered_vehicles/10)* 10
order by 1