# Configuration constants for the traffic simulation and optimization project

# --- Simulation/City Parameters ---
#CITY_NAME = "Bratislava, Slovakia"
#CENTER_COORDS = None#(48.7208, 21.2575)
#RADIUS_KM = None
CITY_NAME = "Košice, Slovakia" 
CENTER_COORDS = (48.7164, 21.2611)
RADIUS_KM = 1
N_VEHICLES = 1000
K_ALTERNATIVES = 3
MIN_LENGTH = 500
MAX_LENGTH = 6000
TIME_STEP = 10
TIME_WINDOW = 300
DIST_THRESH = 10
SPEED_DIFF_THRESH = 2

# --- QUBO/QA Parameters ---
FILTERING_PERCENTAGE = 0.4
LAMBDA_STRATEGY = "standard"#"penalized"   # or "max_weight"
LAMBDA_VALUE = None
COMP_TYPE = "hybrid"             # 'sa', 'hybrid', or 'qpu'
ROUTE_METHOD = "duration"       # or "distance"

# Vehicle origin and destination
ATTRACTION_POINT = None#(48.719390, 21.258057)
D_ALTERNATIVES = None#3

