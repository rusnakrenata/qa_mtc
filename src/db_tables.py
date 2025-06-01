from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Float, JSON, DateTime, BigInteger, Numeric, Boolean, desc, text, Numeric
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime



# Connection string
db_user = "root"
db_password = "Test1234"
db_host = "77.93.155.81"
db_name = "trafficOptimization"

connection_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

# Create SQLAlchemy engine
engine = create_engine(connection_url,
    pool_recycle=280,  # seconds (before wait_timeout)
    pool_pre_ping=True  # check connection before using)
)


# Test connection
try:
    with engine.connect() as connection:
        print("Connection to MariaDB successful!")
except Exception as e:
    print(f"Error connecting to MariaDB: {e}")



""" with engine.connect() as conn:    
    conn.execute(text("DROP TABLE IF EXISTS nodes"))
    conn.execute(text("DROP TABLE IF EXISTS edges")) """

####### --TABLES-- #######

Base = declarative_base()


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
    city_id = Column(Integer, ForeignKey('cities.id'), nullable=False)
    osmid = Column(String(255))
    x = Column(Numeric(9,6))
    y = Column(Numeric(9,6))
    street_count = Column(Integer, nullable=True)
    highway = Column(String(255), nullable=True)
    railway = Column(String(255), nullable=True)
    junction = Column(String(255), nullable=True)    
    geometry = Column(String(255), nullable=True)  # Store as WKT or GeoJSON
    created_at = Column(DateTime, default= datetime.utcnow)

# ---------- Edge Model ----------
class Edge(Base):
    __tablename__ = 'edges'
    
    id = Column(Integer, primary_key=True)
    city_id = Column(Integer, ForeignKey('cities.id'), nullable=False)
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
    city_id = Column(Integer, ForeignKey('cities.id'), nullable=False)
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
    run_configs_id = Column(Integer, ForeignKey('run_configs.id'), nullable=False)  # Link to RunConfig
    iteration_id = Column(Integer, nullable=False) 
    source_edge_id = Column(Integer, ForeignKey('edges.id'))#, related_name='source_edge')
    source_position_on_edge = Column(Float)
    source_geometry = Column(String(255), nullable=True)
    destination_edge_id = Column(Integer, ForeignKey('edges.id'))#, related_name='destination_edge')
    destination_position_on_edge = Column(Float)
    destination_geometry = Column(String(255), nullable=True)
    created_at = Column(DateTime, default= datetime.utcnow)

    #run_configs = relationship("RunConfig", back_populates="vehicles")
    
class VehicleRoute(Base):
    __tablename__ = 'vehicle_routes'

    id = Column(Integer, primary_key=True)
    vehicle_id = Column(Integer, nullable=False) #ForeignKey('vehicles.id'),
    run_configs_id = Column(Integer,  nullable=False)  # Link to RunConfig ForeignKey('run_configs.id'),
    iteration_id = Column(Integer, nullable=False) 
    route_id = Column(Integer, nullable=False) 
    duration = Column(Integer)
    distance = Column(Integer)
    duration_in_traffic = Column(Integer)
    created_at = Column(DateTime, default= datetime.utcnow)

    
class RoutePoint(Base):
    __tablename__ = 'route_points'

    id = Column(Integer, primary_key=True)
    vehicle_id = Column(Integer,  nullable=False) #ForeignKey('vehicles.id'),
    run_configs_id = Column(Integer,  nullable=False)  # Link to RunConfig ForeignKey('run_configs.id'),
    iteration_id = Column(Integer, nullable=False) 
    route_id = Column(Integer, nullable=False) 
    point_id = Column(Integer,  nullable=False) #ForeignKey('vehicle_routes.id'),
    edge_id = Column(Integer, ForeignKey('edges.id'), nullable=False) # closes edge
    cardinal = Column(String(255), nullable=True) # cardinal direction 
    speed = Column(Float)
    lat = Column(Numeric(9,6))
    lon = Column(Numeric(9,6))
    time = Column(Integer)
    created_at = Column(DateTime, default= datetime.utcnow)


class CongestionMap(Base):
    __tablename__ = 'congestion_map'

    id = Column(Integer, primary_key=True)
    run_configs_id = Column(Integer, ForeignKey('run_configs.id'), nullable=False)
    iteration_id = Column(Integer, nullable=False)
    edge_id = Column(Integer, ForeignKey('edges.id'), nullable=False)
    congestion_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    
# class CongestionScore(Base):
#     __tablename__ = 'congestion_scores'

#     id = Column(Integer, primary_key=True)
#     car_id = Column(Integer, ForeignKey('cars.id'), nullable=False)
#     route_index = Column(Integer, nullable=False)
#     run_id = Column(Integer, nullable=False)
#     iteration_id = Column(Integer, nullable=False)
#     score = Column(Float, nullable=False)
#     created_at = Column(DateTime, default=datetime.utcnow)

#     car = relationship("Car", backref="congestion_scores")

# class CongestionWeight(Base):
#     __tablename__ = 'congestion_weights'

#     id = Column(Integer, primary_key=True)
#     run_id = Column(Integer, nullable=False)
#     iteration_id = Column(Integer, nullable=False)
#     car_i_id = Column(Integer, ForeignKey('cars.id'), nullable=False)
#     car_j_id = Column(Integer, ForeignKey('cars.id'), nullable=False)
#     route_index = Column(Integer, nullable=False)
#     weight = Column(Float, nullable=False)
#     created_at = Column(DateTime, default=datetime.utcnow)

#     # Optional: relationships to Car if you want SQLAlchemy backrefs
#     car_i = relationship("Car", foreign_keys=[car_i_id])
#     car_j = relationship("Car", foreign_keys=[car_j_id])


# class SegmentAssignment(Base):
#     __tablename__ = 'segment_assignments'

#     id = Column(Integer, primary_key=True)
#     car_id = Column(Integer, ForeignKey('cars.id'), nullable=False)
#     run_id = Column(Integer, nullable=False)
#     iteration_id = Column(Integer, nullable=False)
#     segment_id = Column(Integer, nullable=False)
#     created_at = Column(DateTime, default=datetime.utcnow)

#     car = relationship("Car")

####### --TABLES-- #######

#Base.metadata.create_all(engine)