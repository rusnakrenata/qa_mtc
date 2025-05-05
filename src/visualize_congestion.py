import folium
from folium import Map, PolyLine, Marker
from branca.colormap import linear

def visualize_routes_by_overlap_congestion(car_routes, congestion_scores):
    if not car_routes:
        print("No routes to display.")
        return None

    # Center map on the first available route
    for car in car_routes:
        if car.get("routes"):
            center = car["origin"]
            break
    else:
        print("No cars have routes.")
        return None

    fmap = folium.Map(location=center, zoom_start=13, tiles='cartodbpositron')

    # Normalize congestion scores for colormap
    all_scores = list(congestion_scores.values())
    colormap = linear.YlOrRd_09.scale(min(all_scores), max(all_scores))
    colormap.caption = "Route Overlap-Based Congestion"
    fmap.add_child(colormap)

    for car in car_routes:
        car_id = car['car_id']
        routes = car.get('routes', [])
        if not routes:
            continue

        # Add start and end markers
        #Marker(location=car['origin'], popup=f"Car {car_id} Start").add_to(fmap)
        #Marker(location=car['destination'], popup=f"Car {car_id} End").add_to(fmap)

        for idx, route in enumerate(routes):
            poly = route['geometry']
            dist_m = route.get('distance', 0)
            time_sec = route.get('duration', 0)
            score = congestion_scores.get((car_id, idx), 0)
            color = colormap(score)

            tooltip_text = (
                f"Car {car_id} - Route {idx}<br>"
                f"Distance: {dist_m / 1000:.2f} km<br>"
                f"Duration: {time_sec // 60:.1f} min<br>"
                f"Congestion Score: {score:.2f}"
            )

            PolyLine(
                locations=poly,
                color=color,
                weight=4,
                opacity=0.7,
                tooltip=tooltip_text
            ).add_to(fmap)

    return fmap
