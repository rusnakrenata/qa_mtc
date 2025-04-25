from db_tables import *
from generate_city_map import store_in_db_city
from generate_cars import store_in_db_cars
from generate_routes import store_in_db_car_routes
from compute_congestion import store_in_db_congestion_scores
from sqlalchemy.orm import sessionmaker

# ---------- CONFIGURATION ----------
API_KEY = 'AIzaSyCawuGvoiyrHOh3RyJdq7yzFCcG5smrZCI'
CITY_NAME = "Košice, Slovakia"
N_CARS = 100
K_ALTERNATIVES = 3
MIN_LENGTH = 200
MAX_LENGTH = 5000

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
    RUN_ID = existing_run.id
    print(f" Run config already exists (run_id={RUN_ID}), skipping insertion.")
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
    RUN_ID = run_config.id
    print(f" Run configuration saved (run_id={RUN_ID}).")


# 3. Store cars for iteration (iteration_is is generated in the cars procedure for each run_config)
CARS, ITERATION_ID = store_in_db_cars(G, N_CARS, MAX_LENGTH, MIN_LENGTH, RUN_ID)

# Store CAR routes in db for the specific iteration ID
CAR_ROUTES = store_in_db_car_routes(CARS, API_KEY, K_ALTERNATIVES, RUN_ID, ITERATION_ID)

# Calculate and store Congestion Scores
CONGESTION_SCORES = store_in_db_congestion_scores(CAR_ROUTES, RUN_ID, ITERATION_ID)

session.close()