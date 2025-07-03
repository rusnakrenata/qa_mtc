# Configuration constants for the traffic simulation and optimization project

# --- Simulation/City Parameters ---
CITY_NAME = "Košice, Slovakia"
N_VEHICLES = 250
K_ALTERNATIVES = 3
MIN_LENGTH = 500
MAX_LENGTH = 6000
TIME_STEP = 60
TIME_WINDOW = 600
DIST_THRESH = 10
SPEED_DIFF_THRESH = 2

# --- QUBO/QA Parameters ---
FILTERING_PERCENTAGE = 0.7
LAMBDA_STRATEGY = "normalized"   # or "max_weight"
LAMBDA_VALUE = N_VEHICLES*(1-FILTERING_PERCENTAGE)/100
COMP_TYPE = "sa"             # 'sa', 'hybrid', or 'qpu'
ROUTE_METHOD = "duration"       # or "distance"
R_VALUE = LAMBDA_VALUE
