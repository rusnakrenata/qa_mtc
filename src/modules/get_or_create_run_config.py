def get_or_create_run_config(session, city_id, RunConfig, nr_vehicles, k_routes, min_length, max_length, time_step, time_window):
    existing_run = session.query(RunConfig).filter_by(
        city_id=city_id,
        n_cars=nr_vehicles,
        k_alternatives=k_routes,
        min_length=min_length,
        max_length=max_length,
        time_step=time_step,
        time_window=time_window
    ).first()

    if existing_run:
        print(f"Run config already exists (run_id={existing_run.id}), skipping insertion.")
        return existing_run
    else:
        run_config = RunConfig(
            city_id=city_id,
            n_cars=nr_vehicles,
            k_alternatives=k_routes,
            min_length=min_length,
            max_length=max_length,
            time_step=time_step,
            time_window=time_window
        )
        session.add(run_config)
        session.commit()
        print(f"Run configuration saved (run_id={run_config.id}).")
        return run_config
