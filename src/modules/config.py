# Configuration constants for the traffic simulation and optimization project

# --- Simulation/City Parameters ---
#CITY_NAME = "Bratislava, Slovakia"
#CENTER_COORDS = None#(48.7208, 21.2575)
#RADIUS_KM = None
CITY_NAME = "Košice, Slovakia"
CENTER_COORDS = (48.7164, 21.2611)
RADIUS_KM = 1.0
N_VEHICLES = 800
K_ALTERNATIVES = 3
MIN_LENGTH = 500
MAX_LENGTH = 6000
TIME_STEP = 10
TIME_WINDOW = 300

# --- QUBO/QA Parameters ---
COMP_TYPE = "hybrid_cqm"             # 'sa', 'hybrid', or 'qpu'
ROUTE_METHOD = "duration"       # or "distance"
MIN_CLUSTER_SIZE = 300
MAX_CLUSTERS = 1

# Vehicle origin and destination
ATTRACTION_POINT = None#(51.4816, -3.1791)
D_ALTERNATIVES = None#3

