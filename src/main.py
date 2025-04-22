from db_tables import *
from city_map import store_in_db_city
from generate_cars import store_in_db_cars
from sqlalchemy.orm import sessionmaker

# ---------- CONFIGURATION ----------
API_KEY = 'AIzaSyCawuGvoiyrHOh3RyJdq7yzFCcG5smrZCI'
CITY_NAME = "Košice, Slovakia"
N_CARS = 1000
K_ALTERNATIVES = 3
MIN_LENGTH = 200
MAX_LENGTH = 2000

Session = sessionmaker(bind=engine)
session = Session()

# 1. Get or create city
G = store_in_db_city(CITY_NAME)
city = session.query(City).filter_by(name=CITY_NAME).first()

# 2. Check for existing identical run config
existing_run = session.query(RunConfig).filter_by(
    city_id=city.id,
    n_cars=N_CARS,
    k_alternatives=K_ALTERNATIVES,
    min_length=MIN_LENGTH,
    max_length=MAX_LENGTH
).first()

if existing_run:
    run_id = existing_run.id
    print(f" Run config already exists (run_id={run_id}), skipping insertion.")
else:
    run_config = RunConfig(
        city_id=city.id,
        n_cars=N_CARS,
        k_alternatives=K_ALTERNATIVES,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH
    )
    session.add(run_config)
    session.commit()
    run_id = run_config.id
    print(f" Run configuration saved (run_id={run_id}).")


# 3. Store cars only if the run is new (or always, if you want to re-generate them)
store_in_db_cars(G, N_CARS, MAX_LENGTH, MIN_LENGTH, run_id)

session.close()