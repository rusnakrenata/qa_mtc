import pickle
import pandas as pd
from database_tables import engine,Base,sessionmaker
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from graph_generation import max_degree
import networkx as nx
import numpy as np



Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

def train_gbr():
    query_for_regression = """
    ## graphs with best partitions and min/max gamma
    with p0 as (
    select g.id as graph_id
    from partitions p
    inner join graphs g on p.graph_id  = g.id
    where nr_of_nodes in (100,200)
    #and edge_probability in (0.1, 0.25, 0.5, 0.75)
    and g.id > 1045
    group by g.id
    having sum(round(lambda_total/lambda_est,4)) >= 0.75
    union all
    select g.id
    from partitions p
    inner join graphs g on p.graph_id  = g.id
    where nr_of_nodes in  (300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000)
    #and edge_probability in (0.1, 0.25, 0.5, 0.75)
    and g.id > 1045
    group by g.id
    having sum(round(lambda_total/lambda_est,4)) >= 0.195 
    ),
    p1 as (
    select min(p.qa_inter_edges) as min_edges,
        p.graph_id
    from p0 
    inner join partitions p on p.graph_id = p0.graph_id
    where 1=1 
    and p.comp_type = "hybrid"
    and p.qa_inter_edges < 1000000
    group by p.graph_id),
    p as (
    select max(p2.lambda_total) as max_lambda_total,
        min(p2.lambda_total) as min_lambda_total,
        p1.graph_id,
        p1.min_edges
    from p1 
    inner join partitions p2 on p1.graph_id = p2.graph_id
                        and p1.min_edges = p2.qa_inter_edges
    where  1=1 
    and p2.qa_inter_edges < 1000000
    group by p1.graph_id, p1.min_edges
    )
    select g.id, g.nr_of_nodes, g.nr_of_edges, 
    g.edge_probability, g.density,
    g.pymetis_inter_edges, g.kernighan_lin_inter_edges,
    p.min_edges, 
    g.lambda_est,p.min_lambda_total, p.max_lambda_total,
    round(p.min_lambda_total / g.lambda_est,3) as min_gamma_coef,
    round(p.max_lambda_total / g.lambda_est,3) as max_gamma_coef,
    case when p.min_edges <= g.pymetis_inter_edges #and p.min_edges <= g.kernighan_lin_inter_edges
    then 1 else 0 end as hybrid_best
    from graphs g
    inner join p on p.graph_id = g.id
    where g.id > 1045
    and g.nr_of_edges is not null 
    and round(p.min_lambda_total / g.lambda_est,3)  is not null
    order by id desc """

    df_for_regression = pd.read_sql(query_for_regression, con=engine)


    # Separate predictors (X) and targets (y)
    X = df_for_regression[['density', 'nr_of_nodes','lambda_est']]
    y_min = df_for_regression['min_gamma_coef']
    y_max = df_for_regression['max_gamma_coef']

    # Split data into training and testing sets
    X_train, X_test, y_min_train, y_min_test, y_max_train, y_max_test = train_test_split(
        X, y_min, y_max, test_size=0.2, random_state=42
    )

    # Initialize Gradient Boosting Regressors
    gbr_min = GradientBoostingRegressor(random_state=42)
    gbr_max = GradientBoostingRegressor(random_state=42)

    # Fit the models
    gbr_min.fit(X_train, y_min_train)
    gbr_max.fit(X_train, y_max_train)

    # Predict intervals
    y_min_pred = gbr_min.predict(X_test)
    y_max_pred = gbr_max.predict(X_test)

    # Combine predictions into an interval
    predicted_intervals = list(zip(y_min_pred, y_max_pred))

    # Display results
    for i, interval in enumerate(predicted_intervals):
        print(f"Sample {i + 1}: Predicted Interval = {interval}")

    # Evaluate for min_value
    mse_min = mean_squared_error(y_min_test, y_min_pred)
    mae_min = mean_absolute_error(y_min_test, y_min_pred)
    rmse_min = np.sqrt(mse_min)
    r2_min = r2_score(y_min_test, y_min_pred)

    print(f"--- min_gamma_coef ---")
    print(f"MAE:  {mae_min:.4f}")
    print(f"MSE:  {mse_min:.4f}")
    print(f"RMSE: {rmse_min:.4f}")
    print(f"R²:   {r2_min:.4f}\n")

    # Evaluate for max_value
    mse_max = mean_squared_error(y_max_test, y_max_pred)
    mae_max = mean_absolute_error(y_max_test, y_max_pred)
    rmse_max = np.sqrt(mse_max)
    r2_max = r2_score(y_max_test, y_max_pred)

    print(f"--- max_gamma_coef ---")
    print(f"MAE:  {mae_max:.4f}")
    print(f"MSE:  {mse_max:.4f}")
    print(f"RMSE: {rmse_max:.4f}")
    print(f"R²:   {r2_max:.4f}")

    # Evaluate for min_value
    mse_min = mean_squared_error(y_min_test, y_min_pred)
    print(f"Mean Squared Error for min_value: {mse_min}")

    # Evaluate for max_value
    mse_max = mean_squared_error(y_max_test, y_max_pred)
    print(f"Mean Squared Error for max_value: {mse_max}")

    with open('gbr_min.pkl', 'wb') as file:
        pickle.dump(gbr_min, file)

    with open('gbr_max.pkl', 'wb') as file:
        pickle.dump(gbr_max, file)


def calculate_lambda(G):
    lambda_est=(1 + min(G.number_of_nodes() / 2 - 1, max_degree(G))) / 2
    X_concrete = pd.DataFrame({
        'density': [nx.density(G)],
        'nr_of_nodes': [G.number_of_nodes()],
        'lambda_est': [lambda_est]
    })

    # Predict lower and upper bounds
    min_pred = gbr_min.predict(X_concrete)
    max_pred = gbr_max.predict(X_concrete)

    return lambda_est*(min_pred+max_pred)/2


train=False

if train:
    train_gbr
else:
    with open('gbr_min.pkl', 'rb') as file:
        gbr_min = pickle.load(file)

    with open('gbr_max.pkl', 'rb') as file:
        gbr_max = pickle.load(file)


