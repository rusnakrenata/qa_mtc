# Configuration constants for the traffic simulation and optimization project

# --- Simulation/City Parameters ---
#CITY_NAME = "Bratislava, Slovakia"
#CENTER_COORDS = None#(48.7208, 21.2575)
#RADIUS_KM = None
CITY_NAME = "Barcelona, Spain"
CENTER_COORDS = (41.392937, 2.145062)#(41.384500, 2.129500) # Barcelona
RADIUS_KM = 3.2
N_VEHICLES = 10000
K_ALTERNATIVES = 3
MIN_LENGTH = 500
MAX_LENGTH = 4000
TIME_STEP = 10
TIME_WINDOW = 300
DISTANCE_FACTOR = 4.0  # Factor to adjust distance in congestion calculations
CLUSTER_RESOLUTION = 4.0  # Resolution for clustering in connectivity analysis

ATTRACTION_POINT = None#(41.380898, 2.122820)#(50.083704,14.432587) # Hlavak
D_ALTERNATIVES = None#4

# --- QUBO/QA Parameters ---
COMP_TYPE = "hybrid"             # 'sa', 'hybrid', or 'qpu'
ROUTE_METHOD = "duration"       # or "distance"
MIN_CLUSTER_SIZE = 500 #400,1000,1100
MAX_CLUSTERS = 200
FULL = False


