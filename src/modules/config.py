# Configuration constants for the traffic simulation and optimization project

# --- Simulation/City Parameters ---
CITY_NAME = "Košice, Slovakia"
N_VEHICLES = 30000
K_ALTERNATIVES = 3
MIN_LENGTH = 1000
MAX_LENGTH = 10000
TIME_STEP = 60
TIME_WINDOW = 600
DIST_THRESH = 10
SPEED_DIFF_THRESH = 2

# --- QUBO/QA Parameters ---
LAMBDA_STRATEGY = "normalized"   # or "max_weight"
LAMBDA_VALUE = 1.0
COMP_TYPE = "sa"             # 'sa', 'hybrid', or 'qpu'
ROUTE_METHOD = "duration"        # or "distance" 