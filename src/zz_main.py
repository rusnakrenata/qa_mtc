from database_tables import Base,engine,sessionmaker
from zz_graph_generation import generate_graph
from pymetis_testing import run_pymetis_partition
from kerninghan_lin_testing import run_kerninghan_lin
from zz_store_into_db import store_graph_to_db, store_partition_to_db
import random


def main(scope=100):
    # Create an SQLite database connection

    # Create all tables
    Base.metadata.create_all(engine)


    # Create a session maker
    Session = sessionmaker(bind=engine)
    #session = Session()

    #QPU Test
    nr_of_nodes_sets = [10,20,30,40,50,60,70,80,90,100]
    edge_probability = [0.1, 0.25, 0.5, 0.75]
    lambda_mults=[0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2]

   
    # QA HS Test
    # p = [0.25, 0,35, 0.5, 0.75, 0.9]
    #lambda_mults=[ 0.1, 0.2, 0.4, 0.6, 0.8, 1] for nr_of_nodes_sets = [10, 30, 50, 80]
    # p = [0.1, 0.25, 0.5, 0.75]
    #lambda_mults=[ 0.05, 0.1, 0.2, 0.4] for nr_of_nodes_sets = [100,200]
    #lambda_mults=[0.005, 0.01, 0.03, 0.05, 0.1] for nr_of_nodes_sets = [300,400,500]
    #lambda_mults=[0.005, 0.01, 0.03, 0.05, 0.1] for nr_of_nodes_sets = [600,700,800,900]
    #lambda_mults=[0.002, 0.005, 0.01, 0.03, 0.05, 0.1] for nr_of_nodes_sets = [1000, 1200, 1400]
    #lambda_mults=[0.002, 0.005, 0.01, 0.03, 0.05, 0.1] for nr_of_nodes_sets = [1600, 1800, 2000]
    #lambda_mults=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.03, 0.05, 0.1] for nr_of_nodes_sets = [2500, 3000, 3500, 4000]
   
    num_reads_default=10


    for i in range(scope):
        with Session() as session:
            print('Graph number:',i)
            nr_of_nodes =  random.choice(nr_of_nodes_sets)
            edge_probability = random.choice(edge_probability)
            graph_data = generate_graph(nr_of_nodes, edge_probability)
            G = graph_data["graph"]
            pymetis_data = run_pymetis_partition(G)
            kl_data = run_kerninghan_lin(G)
            db_graph = store_graph_to_db(session, graph_data, pymetis_data, kl_data, i='qpu', description="qpu")

            comp_type='qpu'
            for lambda_mult in lambda_mults:
                for comp_type in ['hybrid']: 
                    if comp_type in ['qpu','test']:
                        num_reads=num_reads_default
                    else:
                        num_reads=1
                    
                    store_partition_to_db(session=session, G=G, db_graph=db_graph, lambda_mult=lambda_mult*db_graph.lambda_est, comp_type=comp_type, num_reads=num_reads)
            del db_graph


# Check if the script is being run directly
if __name__ == "__main__":
    main()


    






