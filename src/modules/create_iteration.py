def create_iteration(session, run_config_id, provided_iteration_id, Iteration):
    if provided_iteration_id is not None:
        existing = session.query(Iteration).filter_by(id=provided_iteration_id, run_configs_id=run_config_id).first()
        if existing:
            print(f"Run configuration for iteration {provided_iteration_id} already exists. Stopping further execution.")
            return None
        else:
            print("Do not provide iteration_id. It will be generated automatically.")
            return None
    else:
        count = session.query(Iteration).filter_by(run_configs_id=run_config_id).count()
        new_iteration_id = count + 1
        iteration = Iteration(
            iteration_id=new_iteration_id,
            run_configs_id=run_config_id
        )
        session.add(iteration)
        session.commit()
        print(f"Iteration created (iteration_id={new_iteration_id}) for run_config_id={run_config_id}.")
        return new_iteration_id
