# Configuration constants for the traffic simulation and optimization project

# --- Simulation/City Parameters ---
#CITY_NAME = "Bratislava, Slovakia"
#CENTER_COORDS = None#(48.7208, 21.2575)
#RADIUS_KM = None
CITY_NAME = "Košice, Slovakia" 
CENTER_COORDS = None
RADIUS_KM = None
N_VEHICLES = 25000
K_ALTERNATIVES = 3
MIN_LENGTH = 500
MAX_LENGTH = 6000
TIME_STEP = 10
TIME_WINDOW = 300
DIST_THRESH = 10
SPEED_DIFF_THRESH = 2

# --- QUBO/QA Parameters ---
FILTERING_PERCENTAGE = 0.25
LAMBDA_STRATEGY = "normalized"   # or "max_weight"
LAMBDA_VALUE = None
COMP_TYPE = "hybrid"             # 'sa', 'hybrid', or 'qpu'
ROUTE_METHOD = "duration"       # or "distance"

# Vehicle origin and destination
ATTRACTION_POINT = (48.719390, 21.258057)
D_ALTERNATIVES = 5




#CITY_NAME = "London, England" #(covers Central London and much of the congestion charge zone, including Westminster, City of London, Southwark, Camden, etc.)
#CENTER_COORDS = (51.5074, -0.1278)
#RADIUS_KM = 10 #(covers Central London and much of the congestion charge zone, including Westminster, City of London, Southwark, Camden, etc.)
#CITY_NAME = "Paris, France" #(covers Paris and parts of the Île-de-France region)
#CENTER_COORDS = (48.8566, 2.3522)
#RADIUS_KM = 10 #(covers Paris and parts of the Île-de-France region)
#CITY_NAME = "Berlin, Germany" #(covers Berlin and parts of Brandenburg)
#CENTER_COORDS = (52.5200, 13.4050)
#RADIUS_KM = 10 #(covers Berlin and parts of Brandenburg)
