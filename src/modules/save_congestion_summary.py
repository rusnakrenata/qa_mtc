from typing import Any
import pandas as pd
from models import CongestionSummary  # Make sure this import matches your project structure

def save_congestion_summary(
    session: Any,
    run_config_id: int,
    iteration_id: int,
    edges_df: pd.DataFrame,
    congestion_df: pd.DataFrame,
    post_qa_congestion_df: pd.DataFrame,
    shortest_routes_dur_df: pd.DataFrame,
    shortest_routes_dis_df: pd.DataFrame,
    random_routes_df: pd.DataFrame,
    post_gurobi_df: pd.DataFrame,
) -> None:
    """Save congestion summary to the database."""

    # 1. Aggregate base congestion
    congestion_df_grouped = congestion_df.groupby('edge_id', as_index=False).agg({'congestion_score': 'sum'})

    # 2. Initialize merged DataFrame with all edge IDs
    merged = pd.DataFrame({'edge_id': edges_df.drop_duplicates(subset='edge_id')['edge_id']})

    # 3. Merge in each congestion type
    merged = merged.merge(
        congestion_df_grouped.rename(columns={'congestion_score': 'congestion_all'}), on='edge_id', how='left')
    merged = merged.merge(
        post_qa_congestion_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_post_qa'}),
        on='edge_id', how='left')
    merged = merged.merge(
        shortest_routes_dur_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_shortest_dur'}),
        on='edge_id', how='left')
    merged = merged.merge(
        shortest_routes_dis_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_shortest_dis'}),
        on='edge_id', how='left')
    merged = merged.merge(
        random_routes_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_random'}),
        on='edge_id', how='left')
    merged = merged.merge(
        post_gurobi_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_post_gurobi'}),
        on='edge_id', how='left')

    # 4. Fill missing values with 0
    merged = merged.fillna(0)

    # 5. Convert each row to a CongestionSummary record
    records = [
        CongestionSummary(
            run_configs_id=run_config_id,
            iteration_id=iteration_id,
            edge_id=int(row['edge_id']),
            congestion_all=float(row['congestion_all']),
            congestion_post_qa=float(row['congestion_post_qa']),
            congestion_post_sa=0.0,
            congestion_post_tabu=0.0,
            congestion_shortest_dur=float(row['congestion_shortest_dur']),
            congestion_shortest_dis=float(row['congestion_shortest_dis']),
            congestion_random=float(row['congestion_random']),
            congestion_post_gurobi=float(row['congestion_post_gurobi']),
            congestion_post_cbc=0.0
        )
        for _, row in merged.iterrows()
    ]

    # 6. Store to DB
    try:
        session.add_all(records)
        session.commit()
    except Exception as e:
        session.rollback()
        raise RuntimeError("Failed to commit congestion summary to the database.") from e
    finally:
        session.close()
