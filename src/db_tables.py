from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Float, JSON, DateTime, BigInteger
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
    name = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    cars = relationship("Car", back_populates="city")

class Car(Base):
    __tablename__ = 'cars'
    
    id = Column(Integer, primary_key=True)
    city_id = Column(Integer, ForeignKey('cities.id'), nullable=False)
    src_node = Column(BigInteger)
    dst_node = Column(BigInteger)
    src_lat = Column(Float)
    src_lon = Column(Float)
    dst_lat = Column(Float)
    dst_lon = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    city = relationship("City", back_populates="cars")






####### --TABLES-- #######

Base.metadata.create_all(engine)