# Configuration constants for the traffic simulation and optimization project

# --- Simulation/City Parameters ---
CITY_NAME = "Košice, Slovakia"
CENTER_COORDS = None#(48.7208, 21.2575)
N_VEHICLES = 250
K_ALTERNATIVES = 3
MIN_LENGTH = 500
MAX_LENGTH = 6000
TIME_STEP = 5
TIME_WINDOW = 300
DIST_THRESH = 10
SPEED_DIFF_THRESH = 2

# --- QUBO/QA Parameters ---
FILTERING_PERCENTAGE = 0.25
LAMBDA_STRATEGY = "normalized"   # or "max_weight"
LAMBDA_VALUE = None
COMP_TYPE = "sa"             # 'sa', 'hybrid', or 'qpu'
ROUTE_METHOD = "duration"       # or "distance"

