# Maxspots Curvature

![](https://github.com/nda-github/maxspots_curvature/blob/master/images/curvature_map_geophysics.JPG "Curvature lineaments")

# What is it?

### This program automates lineament detection in point data for geophysical analysis (see Phillips et al., 2007, The use of curvature in potential-field interpretation) 

### The algorithm works as follows:
#### 1. Upscale the input data to a grid with cell size equal to the distance tolerance.
#### 2. Select a random unvisited point.
#### 3. Grow the line by continually selecting the next best point from the previous point's cell neighborhood (given a specified distance tolerance and azimuth tolerance) until there are no more acceptable points. 
#### 4. Reverse the line path, and grow the line in the other direction. Mark each point as visited.
#### 5. Repeat from step 2 until there are no more points to visit.

# What does it look like?
![](https://github.com/nathens/maxspots_curvature/blob/master/images/map_zoom.JPG "Closeup input/output")
![](https://github.com/nda-github/maxspots_curvature/blob/master/images/map.JPG "Maxspots output")