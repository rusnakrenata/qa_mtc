import matplotlib.pyplot as plt

def plot_congestion_heatmap(edges_gdf, congestion_df, cmap='Reds', figsize=(12, 12)):
    """
    Aggregates congestion data over edges and plots a heatmap.

    Parameters:
    - edges_gdf: GeoDataFrame containing road network edges with geometry.
    - congestion_df: DataFrame containing edge_id and congestion_score.
    - cmap: Color map used for the heatmap (default: 'Reds').
    - figsize: Tuple defining the size of the plot (default: (12, 12)).
    """
    if congestion_df is None or congestion_df.empty:
        print("No congestion map data to plot.")
        return

    # Merge and clean
    merged_gdf = edges_gdf.merge(congestion_df, left_on='id', right_on='edge_id', how='left')
    merged_gdf['congestion_score'] = merged_gdf['congestion_score'].fillna(0)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    merged_gdf.plot(
        column='congestion_score',
        cmap=cmap,
        linewidth=2,
        ax=ax,
        legend=True,
        legend_kwds={'label': "Congestion Score", 'shrink': 0.5}
    )
    ax.set_title('Traffic Congestion Heatmap')
    ax.set_axis_off()
    plt.show()
