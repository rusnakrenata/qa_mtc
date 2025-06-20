from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Float, JSON, DateTime, BigInteger, Numeric, Boolean, desc, text, Numeric, Index, PrimaryKeyConstraint
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime



# Connection string
db_user = "traffic_opti"
db_password = "P4ssw0rd"
db_host = "147.232.204.254"#"77.93.155.81"
db_name = "trafficOptimization"

#connection_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
connection_url = f"mysql+mysqldb://{db_user}:{db_password}@{db_host}/{db_name}"

# Create SQLAlchemy engine
engine = create_engine(connection_url,
    pool_recycle=280,  # seconds (before wait_timeout)
    pool_pre_ping=True  # check connection before using)
)


# Test connection
try:
    with engine.connect() as connection:
        Base = declarative_base()
except Exception as e:
    print(f"Error connecting to MariaDB: {e}")



####### --TABLES-- #######

class City(Base):
    __tablename__ = 'cities'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    node_count = Column(Integer)
    edge_count = Column(Integer)
    created_at = Column(DateTime, default= datetime.utcnow)

    #run_configs = relationship("RunConfig", back_populates="city")

# ---------- Node Model ----------
class Node(Base):
    __tablename__ = 'nodes'
    
    id = Column(Integer, primary_key=True)
    city_id = Column(Integer,  nullable=False)
    osmid = Column(String(255))
    x = Column(Numeric(9,6))
    y = Column(Numeric(9,6))
    geometry = Column(String(255), nullable=True)  # Store as WKT or GeoJSON
    created_at = Column(DateTime, default= datetime.utcnow)

# ---------- Edge Model ----------
class Edge(Base):
    __tablename__ = 'edges'
    
    id = Column(Integer, primary_key=True)
    city_id = Column(Integer,  nullable=False)
    #osmid = Column(Integer, nullable=False, unique=True)
    # Removed u, v columns, no longer representing nodes
    u = Column(String(255))
    v = Column(String(255))
    length = Column(String(255), nullable=True)
    geometry = Column(String(10000), nullable=True)  # Store as GeoJSON or WKT format for simplicity
    created_at = Column(DateTime, default= datetime.utcnow)

class Iteration(Base):
    __tablename__ = 'iterations'

    id = Column(Integer, primary_key=True)
    iteration_id = Column(Integer, nullable=False)
    run_configs_id = Column(Integer)  # Link to RunConfig
    created_at = Column(DateTime, default= datetime.utcnow)

    #city = relationship("City", back_populates="iterations")

class RunConfig(Base):
    __tablename__ = 'run_configs'

    id = Column(Integer, primary_key=True)
    city_id = Column(Integer,  nullable=False)
    n_cars = Column(Integer)
    k_alternatives = Column(Integer)
    min_length = Column(Integer)
    max_length = Column(Integer)
    time_step = Column(Integer, nullable=False)
    time_window = Column(Integer, nullable=False)
    created_at = Column(DateTime, default= datetime.utcnow)

    #city = relationship("City", back_populates="run_configs")
    

class Vehicle(Base):
    __tablename__ = 'vehicles'

    id = Column(Integer, primary_key=True)
    vehicle_id = Column(BigInteger, nullable=False) 
    run_configs_id = Column(Integer,  nullable=False)  # Link to RunConfig
    iteration_id = Column(Integer, nullable=False) 
    origin_edge_id = Column(Integer, nullable=False)#, related_name='origin_edge')
    origin_position_on_edge = Column(Float)
    origin_geometry = Column(String(255), nullable=True)
    destination_edge_id = Column(Integer, nullable=False)#, related_name='destination_edge')
    destination_position_on_edge = Column(Float)
    destination_geometry = Column(String(255), nullable=True)
    created_at = Column(DateTime, default= datetime.utcnow)

    #run_configs = relationship("RunConfig", back_populates="vehicles")
    
class VehicleRoute(Base):
    __tablename__ = 'vehicle_routes'

    #id = Column(Integer, autoincrement=True, nullable=False)
    vehicle_id = Column(Integer, nullable=False) #ForeignKey('vehicles.id'),
    run_configs_id = Column(Integer,  nullable=False)  # Link to RunConfig ForeignKey('run_configs.id'),
    iteration_id = Column(Integer, nullable=False) 
    route_id = Column(Integer, nullable=False) 
    duration = Column(Integer)
    distance = Column(Integer)
    duration_in_traffic = Column(Integer)
    created_at = Column(DateTime, default= datetime.utcnow)

    __table_args__ = (
        PrimaryKeyConstraint('run_configs_id', 'iteration_id', 'vehicle_id', 'route_id'),
        Index('idx_run_iter_vehicle_method', 'run_configs_id', 'iteration_id', 'vehicle_id')
    )


    
class RoutePoint(Base):
    __tablename__ = 'route_points'

    #id = Column(Integer, autoincrement=True, nullable=False)
    vehicle_id = Column(Integer, nullable=False)
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    route_id = Column(Integer, nullable=False)
    point_id = Column(Integer, nullable=False)
    edge_id = Column(Integer, nullable=False)
    cardinal = Column(String(255), nullable=True)
    speed = Column(Float)
    lat = Column(Numeric(9, 6))
    lon = Column(Numeric(9, 6))
    time = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        # Composite Primary Key - OK as-is, used for uniqueness and integrity
        PrimaryKeyConstraint(
            'run_configs_id', 'iteration_id', 
            'edge_id', 'vehicle_id', 'route_id', 'point_id', 'time' ),
        Index('edge_id', 'time', 'cardinal', 'vehicle_id','lat','lon')

    )
    

class CongestionMap(Base):
    __tablename__ = 'congestion_map'

    id = Column(Integer, primary_key=True)
    run_configs_id = Column(Integer,  nullable=False)
    iteration_id = Column(Integer, nullable=False)
    edge_id = Column(Integer,  nullable=False)
    vehicle1 = Column(Integer,  nullable=False)
    vehicle1_route = Column(Integer, nullable=False)
    vehicle2 = Column(Integer,  nullable=False)
    vehicle2_route = Column(Integer, nullable=False)
    congestion_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class QAResult(Base):
    __tablename__ = 'qa_results'

    id = Column(Integer, primary_key=True)
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    lambda_strategy = Column(String(50))
    lambda_value = Column(Float)
    comp_type = Column(String(50))
    num_reads = Column(Integer)
    n_vehicles = Column(Integer)
    k_alternattives = Column(Integer)  # follows your return dict spelling
    weights = Column(JSON)             # serialized congestion weight matrix
    vehicle_ids = Column(JSON)         # list of filtered vehicle IDs
    assignment_valid = Column(Integer) # 1 or 0
    assignment = Column(JSON)          # route assignments for each vehicle
    energy = Column(Float)
    duration = Column(Float)
    qubo_path = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    

    
####### --TABLES-- #######

Base.metadata.create_all(engine)