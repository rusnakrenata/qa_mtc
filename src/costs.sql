
SET @dist_thresh := 10.0;
SET @speed_diff_thresh := 5.0;
SET @iteration := 11;

-- Step 1: All valid pairs in the same group

 with allpairs as (
 Select 
     a.vehicle_id AS vehicle_id_1,
    b.vehicle_id AS vehicle_id_2,
    a.route_id as route_id_1,
    b.route_id as route_id_2,
    a.duration+b.duration as duration,
    a.distance+b.distance as distance
 FROM trafficOptimization.vehicle_routes a
 JOIN trafficOptimization.vehicle_routes b
 on a.vehicle_id < b.vehicle_id
 where a.iteration_id=@iteration and b.iteration_id=@iteration
 ),

 minvalues as
(SELECT  vehicle_id,iteration_id,min(distance) as min_distance,min(duration) as min_duration 
FROM trafficOptimization.vehicle_routes
where iteration_id=@iteration
group by vehicle_id), 

allpairsminvalues as 
(Select allpairs.*,a.min_distance+b.min_distance as min_distance,
a.min_duration+b.min_duration as min_duration from allpairs 
join minvalues as a on a.vehicle_id=allpairs.vehicle_id_1 
join minvalues as b on b.vehicle_id=allpairs.vehicle_id_2 
),



pairwise AS (
SELECT
    a.time,
    a.edge_id,
    a.cardinal,
    a.vehicle_id AS vehicle_id_1,
    b.vehicle_id AS vehicle_id_2,
    a.route_id as route_id_1,
    b.route_id as route_id_2,
    a.speed AS speed_1,
    b.speed AS speed_2,
    6371 * 2 * ASIN(SQRT(
    POW(SIN(RADIANS(b.lat - a.lat) / 2), 2) +
    COS(RADIANS(a.lat)) * COS(RADIANS(b.lat)) *
    POW(SIN(RADIANS(b.lon - a.lon) / 2), 2)
  )) AS distance,
    ABS(a.speed - b.speed) AS speed_diff
FROM trafficOptimization.route_points a
 JOIN trafficOptimization.route_points b
  ON a.time = b.time
 AND a.edge_id = b.edge_id
 AND a.cardinal = b.cardinal
 AND a.vehicle_id < b.vehicle_id
 where a.iteration_id=@iteration and b.iteration_id=@iteration)
 

Select allpairsminvalues.vehicle_id_1,allpairsminvalues.route_id_1,allpairsminvalues.vehicle_id_2,allpairsminvalues.route_id_2,sum(case when pairwise.distance < @dist_thresh AND speed_diff < @speed_diff_thresh then 1 else 0 end) as congestion , sum(duration-min_duration) as delta_duration, sum(allpairsminvalues.distance-min_distance) as delta_distance
from 
allpairsminvalues
left JOIN pairwise on (allpairsminvalues.vehicle_id_1=pairwise.vehicle_id_1 and allpairsminvalues.vehicle_id_2=pairwise.vehicle_id_2 and allpairsminvalues.route_id_1=pairwise.route_id_1 and allpairsminvalues.route_id_2=pairwise.route_id_2)

 
group by allpairsminvalues.vehicle_id_1,allpairsminvalues.vehicle_id_2,allpairsminvalues.route_id_1, allpairsminvalues.route_id_2;

