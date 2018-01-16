import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.cluster.hierarchy as hac
import numpy as np
import dijkstra as dij

pd.options.mode.chained_assignment = None

CURVATURE_FILE = 'Grav_ALL_200_HGM_MS.csv'
CURVATURE_PATH = 'datasets'
MIN_POINTS = 3 # Must be greater than 3
AZIMUTH_TOL = 40 # Degrees
DISTANCE_TOL = None #

def load_curvature_file(curvature_file=CURVATURE_FILE, curvature_path=CURVATURE_PATH):
    csv_path = os.path.join(curvature_path, curvature_file)
    return pd.read_csv(csv_path)

def dist_nearest_point(points):
    dists = sp.spatial.distance.pdist(points)
    dists_matrix = sp.spatial.distance.squareform(dists)
    dists_matrix[dists_matrix == 0] = np.nan
    return np.nanmin(dists_matrix, axis = 0)

def calculate_distance_tolerance(df):
    dist_to_nearest = dist_nearest_point(df[['X', 'Y']])
    return np.mean(dist_to_nearest) + np.std(dist_to_nearest)

def hac_clustering(df, method, height):
    Z = hac.linkage(df[['X', 'Y']].values, method = method)
    return hac.cut_tree(Z, height = height)

def remove_small_clusters(df, channel = 'clustering', min_points = 3):
    unique, counts = np.unique(df[channel], return_counts = True)
    lines_with_points = unique[counts > (min_points - 1)]
    return df[df[channel].isin(lines_with_points)]

def get_farthest_point(points):
    center = np.mean(points, axis=0)
    dist_to_center = np.linalg.norm(points - center, axis = 1)
    return points.iloc[np.argmax(dist_to_center)]

def points_to_graph(points, dist_threshold):
    graph = {}
    for index, row in points.iterrows():
        points_to_check = points.copy()
        current_point = points_to_check.loc[index]
        points_to_check.drop(index, inplace = True)
        dists = np.linalg.norm(points_to_check[['X', 'Y']] - current_point[['X', 'Y']], axis = 1)
        neighbors = points_to_check.iloc[np.argsort(dists)]
        neighbors['dists'] = dists
        neighbors = neighbors[np.linalg.norm(neighbors[['X', 'Y']] - current_point[['X', 'Y']], axis = 1) < dist_threshold]
        graph[index] = neighbors[['dists']].T.to_dict('records')[0]
    return graph

def calc_azimuth(point1, point2):
    """ Returns heading in degrees clockwise from north """
    dx = float(point2[0] - point1[0])
    dy = float(point2[1] - point1[1])
    if dx > 0: theta = (np.pi*0.5) - np.arctan(dy/dx)
    elif dx < 0: theta = (np.pi*1.5) - np.arctan(dy/dx)
    elif dy > 0: theta = 0
    elif dy < 0: theta = np.pi
    else:
        theta = 99999
    return np.degrees(theta)

def azimuth_difference(azimuth1, azimuth2):
    """ Returns difference of two azimuths in degrees """
    azimuth_diff = np.absolute(azimuth1 - azimuth2)
    if azimuth_diff > 180: azimuth_diff = 360 - azimuth_diff
    return azimuth_diff

def select_optimal_path(paths, points):
    paths = sorted(paths, key = len, reverse = True) # Look at longest path first
    azimuth_check = False
    while (azimuth_check == False):
        for path in paths:
            if len(path) > 0:
                azimuth_deviation = []
                for i in range(0, len(path) - 2):
                    az1 = calc_azimuth(points.loc[path[i]][['X', 'Y']], points.loc[path[i + 1]][['X', 'Y']])
                    az2 = calc_azimuth(points.loc[path[i + 1]][['X', 'Y']], points.loc[path[i + 2]][['X', 'Y']])
                    azimuth_deviation.append(azimuth_difference(az1, az2))
                if all(az < AZIMUTH_TOL for az in azimuth_deviation):
                    azimuth_check = True
                    break
            else:
                azimuth_check = True
                break
    return path

def update_points(points, points_assigned, path, line):
    for id_ in path:
        next_row = points_assigned[pd.isnull(points_assigned.X)].index.min()
        next_point = points.loc[id_][['X', 'Y']]
        next_point['line'] = line
        next_point['id'] = points.loc[id_].name
        points_assigned.loc[next_row] = next_point
        points.drop(id_, inplace = True)
    return points, points_assigned

def assign_lines(points, dist_threshold):
    all_points = points.copy()
    points_assigned = pd.DataFrame(index = range(points.shape[0]), columns = points.columns, dtype='float')
    points_assigned['line'] = np.nan
    points_assigned['id'] = np.nan
    min_distance = dist_nearest_point(points).min()
    line = 1
    while (min_distance < dist_threshold):
        start_point = get_farthest_point(points)
        graph = points_to_graph(points, dist_threshold)
        paths = [dij.shortestPath(graph, start_point.name, x) for x in points.index][1:]
        path = select_optimal_path(paths, points)
        points, points_assigned = update_points(points, points_assigned, path, line)
        if (start_point.name in points.index): points.drop(start_point.name, inplace = True)
        line = line + 1
        min_distance = dist_nearest_point(points).min()
    points_assigned.dropna(axis=0, how='any', inplace=True)
    points_assigned[['line', 'id']] = points_assigned[['line', 'id']].astype('int')
    return points_assigned

def clusters_to_lines(df, dist_threshold):
    if DISTANCE_TOL:
        dist_threshold = DISTANCE_TOL
    else:
        dist_threshold = dist_threshold * 1 # Tuning parameter to force lines to go through more points
    df['Line'] = 0
    df['FID'] = 0
    i = 1
    print df.clustering.max()
    for cluster in df.clustering.unique():
        print cluster
        cluster_points = df[df.clustering == cluster][['X', 'Y']]
        lines = assign_lines(cluster_points, dist_threshold)
        lines.line = lines.line + df.Line.max()
        for id_ in lines.id:
            df.loc[id_, 'Line'] = int(lines[lines.id == id_].line)
            df.loc[id_, 'FID'] = i
            i = i + 1
    return df

def plot_lines(original, df, show = False):
    plt.figure()
    plt.plot(original.X, original.Y, 'k.')
    for line in df.Line.unique():
        points = df[df.Line == line][['X', 'Y']]
        plt.plot(points.X, points.Y, '-')
    plt.axis('equal')
    if show: plt.show()
    return None

def main():
    df = load_curvature_file()
    #dist_threshold = calculate_distance_tolerance(df)
    #df['clustering'] = hac_clustering(df, 'single', dist_threshold)
    #df = remove_small_clusters(df, min_points = 3) # 3 is minimum input
    #df.to_pickle('TestSet')

    #df = pd.read_pickle('TestSet')
    #df = clusters_to_lines(df, dist_threshold = 272)
    #df.to_pickle('TestSetLines')

    original = df.copy()
    df = pd.read_pickle('TestSetLines')

    df = df[df.Line > 0]
    df = df.sort_values(['FID'])
    df = remove_small_clusters(df, 'Line', min_points = 3)

    plot_lines(original, df, show = True)

    #print df.head(50)

    #df = df.sort_values(['FID'])
    #print df.head(10)
    #print df[df.Line == 14]


if __name__=='__main__':
    main()
