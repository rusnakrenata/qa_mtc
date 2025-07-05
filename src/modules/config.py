# Configuration constants for the traffic simulation and optimization project

# --- Simulation/City Parameters ---
CITY_NAME = "Košice, Slovakia"
N_VEHICLES = 25000
K_ALTERNATIVES = 3
MIN_LENGTH = 500
MAX_LENGTH = 6000
TIME_STEP = 5
TIME_WINDOW = 300
DIST_THRESH = 10
SPEED_DIFF_THRESH = 2

# --- QUBO/QA Parameters ---
FILTERING_PERCENTAGE = 0.7
LAMBDA_STRATEGY = "normalized"   # or "max_weight"
LAMBDA_VALUE = 1#N_VEHICLES*(1-FILTERING_PERCENTAGE)/100
COMP_TYPE = "hybrid"             # 'sa', 'hybrid', or 'qpu'
ROUTE_METHOD = "duration"       # or "distance"
R_VALUE = 0#N_VEHICLES*(1-FILTERING_PERCENTAGE)/100
