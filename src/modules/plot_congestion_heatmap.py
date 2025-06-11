import matplotlib.pyplot as plt
import folium
import branca.colormap as cm
from branca.colormap import LinearColormap, linear
from shapely.geometry import LineString
import geopandas as gpd

def plot_congestion_heatmap_interactive(edges_gdf, congestion_df, offset_deg=0.00005):
    """
    Plot interactive congestion heatmap using Folium with green-to-red color scale and tooltips.
    """
    if congestion_df is None or congestion_df.empty:
        print("No congestion map data to plot.")
        return
    
    # Merge scores
    merged = edges_gdf.merge(congestion_df, left_on='id', right_on='edge_id', how='left')
    merged['congestion_score'] = merged['congestion_score'].fillna(0)
    merged['edge_id'] = merged['id']

    # Convert to WGS84
    merged = merged.to_crs(epsg=4326)

    # Create folium map centered on mean geometry centroid
    minx, miny, maxx, maxy = merged.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='cartodbpositron')
    m.fit_bounds([[miny, minx], [maxy, maxx]])  # Ensures all edges are visible


    vmin = merged['congestion_score'].min()
    vmax = merged['congestion_score'].max()

    colormap = LinearColormap(
        ['silver', 'yellow','red','purple'], 
        vmin=vmin,
        vmax=vmax,
        caption='Congestion Score (Green → Red)'
    )

    colormap.add_to(m)
   
    # Plot each edge as a polyline
    for _, row in merged.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or geom.length == 0:
                continue

            # Use direction to choose offset side
            coords = list(geom.coords)
            x0, y0 = coords[0]
            x1, y1 = coords[-1]

            # If going south or west, offset to left; else to right
            if ((y1 < y0) and (x1 < x0)) or ((y1 > y0) and (x1 > x0)):
                side = 'left'
            else:
                side = 'right'

            try:
                offset_geom = geom.parallel_offset(offset_deg, side=side, join_style=2)
                if isinstance(offset_geom, LineString):
                    lines = [offset_geom]
                else:
                    lines = list(offset_geom.geoms)

                for line in lines:
                    coords = [(lat, lon) for lon, lat in line.coords]
                    folium.PolyLine(
                        coords,
                        color=colormap(row['congestion_score']),
                        weight=3,
                        opacity=0.7,
                        tooltip=f"Edge ID: {row['id']}<br>Score: {row['congestion_score']:.2f}"
                    ).add_to(m)
            except Exception as e:
                print(f"Skipping edge {row['id']} due to offset error: {e}")

    return m



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


