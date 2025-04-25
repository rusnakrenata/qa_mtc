from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Float, JSON, DateTime, BigInteger, Numeric, Boolean, desc
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime



# Connection string
db_user = "root"
db_password = "Test1234"
db_host = "77.93.155.81"
db_name = "trafficOptimization"

connection_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

# Create SQLAlchemy engine
engine = create_engine(connection_url)

# Test connection
try:
    with engine.connect() as connection:
        print("Connection to MariaDB successful!")
except Exception as e:
    print(f"Error connecting to MariaDB: {e}")


####### --TABLES-- #######

Base = declarative_base()

class City(Base):
    __tablename__ = 'cities'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    node_count = Column(Integer)
    edge_count = Column(Integer)
    created_at = Column(DateTime, default= datetime.utcnow)

    run_configs = relationship("RunConfig", back_populates="city")
    traffic_lights = relationship("TrafficLight", back_populates="city")

class TrafficLight(Base):
    __tablename__ = 'traffic_lights'

    id = Column(Integer, primary_key=True)
    city_id = Column(Integer, ForeignKey('cities.id'), nullable=False)
    lat = Column(Float)
    lon = Column(Float)
    red_cycle = Column(Integer, default=30)  # Estimated red light wait time in seconds
    created_at = Column(DateTime, default= datetime.utcnow)

    city = relationship("City", back_populates="traffic_lights")

class Car(Base):
    __tablename__ = 'cars'

    id = Column(Integer, primary_key=True)
    car_id = Column(BigInteger, nullable=False)
    run_configs_id = Column(Integer, ForeignKey('run_configs.id'), nullable=False)  # Link to RunConfig
    iteration_id = Column(Integer, nullable=False)
    src_node = Column(BigInteger)
    dst_node = Column(BigInteger)
    src_lat = Column(Numeric(9, 6))  # 6 decimal places
    src_lon = Column(Numeric(9, 6))
    dst_lat = Column(Numeric(9, 6))
    dst_lon = Column(Numeric(9, 6))
    created_at = Column(DateTime, default= datetime.utcnow)

    run_configs = relationship("RunConfig", back_populates="cars")
    
class RunConfig(Base):
    __tablename__ = 'run_configs'

    id = Column(Integer, primary_key=True)
    city_id = Column(Integer, ForeignKey('cities.id'), nullable=False)
    n_cars = Column(Integer)
    k_alternatives = Column(Integer)
    min_length = Column(Integer)
    max_length = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    city = relationship("City", back_populates="run_configs")
    cars = relationship("Car", back_populates="run_configs")
    
class CarRoute(Base):
    __tablename__ = 'car_routes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    car_id = Column(Integer, ForeignKey('cars.id'), nullable=False)
    iteration_id = Column(Integer, nullable=False)
    route_index = Column(Integer, nullable=False)
    geometry = Column(JSON)  # list of lat/lon pairs
    duration = Column(Integer)
    distance = Column(Integer)
    duration_in_traffic = Column(Integer)
    traffic_light_count = Column(Integer)
    total_red_light_wait = Column(Float)
    contains_traffic_light = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)

    car = relationship("Car", backref="car_routes")


class CongestionScore(Base):
    __tablename__ = 'congestion_scores'

    id = Column(Integer, primary_key=True)
    car_id = Column(Integer, ForeignKey('cars.id'), nullable=False)
    route_index = Column(Integer, nullable=False)
    run_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    car = relationship("Car", backref="congestion_scores")

class CongestionWeight(Base):
    __tablename__ = 'congestion_weights'

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    car_i_id = Column(Integer, ForeignKey('cars.id'), nullable=False)
    car_j_id = Column(Integer, ForeignKey('cars.id'), nullable=False)
    route_index = Column(Integer, nullable=False)
    weight = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Optional: relationships to Car if you want SQLAlchemy backrefs
    car_i = relationship("Car", foreign_keys=[car_i_id])
    car_j = relationship("Car", foreign_keys=[car_j_id])



####### --TABLES-- #######

Base.metadata.create_all(engine)