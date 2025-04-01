from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Float, JSON
from sqlalchemy.orm import declarative_base, relationship, sessionmaker


# Connection string
db_user = "user"
db_password = "password"
db_host = "localhost"
db_name = "graphPartitioning"

connection_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

# Create SQLAlchemy engine
engine = create_engine(connection_url)

# Test connection
try:
    with engine.connect() as connection:
        print("Connection to MariaDB successful!")
except Exception as e:
    print(f"Error connecting to MariaDB: {e}")

Base = declarative_base()
# Graphs Table
class Graph(Base):
    __tablename__ = 'graphs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text)
    nr_of_nodes = Column(Integer) 
    edge_probability = Column(Float)
    edges = Column(JSON)  # Store list of nodes as JSON
    pymetis_partition_nodes = Column(JSON, nullable=True)
    pymetis_inter_edges = Column(Integer, nullable=True)
    pymetis_intra_edges = Column(Integer, nullable=True)
    lambda_est = Column(Float, nullable=True)
    kernighan_lin_inter_edges = Column(Integer, nullable=True)
    density = Column(Float, nullable=True)
    clustering = Column(Float, nullable=True)
    nr_of_edges=Column(Integer, nullable=True)
    duration_pymetis = Column(Float, nullable=True)
    duration_keringham_lin = Column(Float, nullable=True)

    
    # Relationship to Partition table
    partitions = relationship("Partition", back_populates="graph", cascade="all, delete-orphan")


# Partitions Table
class Partition(Base):
    __tablename__ = 'partitions'
    # Removed autoincrement from 'id' as composite key
    id = Column(Integer, primary_key=True)  # Keep it a normal column, no autoincrement
    graph_id = Column(Integer, ForeignKey('graphs.id'), primary_key=True)  # ForeignKey referring to 'graphs.id'
    comp_type = Column(Text, nullable=True)
    nodes = Column(Text, nullable=True)  # Store list of nodes as JSON
    qa_inter_edges = Column(Integer, nullable=True)
    lambda_mult = Column(Float, nullable=True)
    lambda_total = Column(Float, nullable=True)
    duration_qa = Column(Float, nullable=True)
    duration_full = Column(Float, nullable=True) # qa + matrix calc

    # Relationship back to Graph
    graph = relationship("Graph", back_populates="partitions")
