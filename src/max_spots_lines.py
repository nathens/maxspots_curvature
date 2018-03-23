# max_spots_lines.py
# Created: March 20th, 2018

"""
This interactive script organizes point data into lines, intended to be used on
curvature extrema point data (see Phillips et al., 2007, The use of
curvature in potential-field interpretation)
"""
import sys
import pandas as pd
import numpy as np
from random import shuffle, sample
import os.path
from Queue import Queue, PriorityQueue
pd.options.mode.chained_assignment = None  # default='warn'

def update_progress(part, total):
    """Displays progress in percent """
    progress = 100 * (float(part) / total)
    sys.stdout.write("\rProgram progress:  %d%%" % progress)
    sys.stdout.flush()
    return None

def welcome_message():
    """ Prints welcome message to user """
    print '******************************************************************************* \n'
    print '******************************* CURVATURE LINES ******************************* \n'
    print '******************************************************************************* \n'
    print '@author Noah Athens \n'
    print '@repository https://github.com/nathens/maxspots_curvature\n'
    print '@version 3/22/2018 \n'
    print 'This program organizes point data resulting from the usgs_curv4.gx into lines. \n'
    print 'To use this program, export the curvature database as a CSV file \n'
    print '(including header). Coordinates must be projected and labeled "X" and "Y".\n'
    print 'Example header: ID,Z1,Z2,Z3,Z4,Z5,Z6,X,Y,X2,Y2,__X,__Y\n'
    return None

def user_end_action():
    """ Print completed message and prompt user """
    print '\n'
    print 'Program completed. Output file exported to current directory \n\n'
    answer = raw_input("Process another file? [y/n]: ")
    return answer[0].lower() == 'y'

def prompt_user_for_number(prompt):
    """ Prompts user for valid numeric input """
    while True:
        num = raw_input(prompt)
        try:
            float(num)
            break
        except ValueError:
            print '\n' + num + ' is not a valid number.\n'
    return float(num)

def prompt_user_for_file(prompt):
    """ Prompts user for valid filename """
    while True:
        fname = raw_input(prompt)
        if os.path.isfile(fname):
            break
        else: print '\n' + fname + ' is not a valid file.\n'
    return fname

def get_inputs_from_user(nearest):
    """ Returns inputs from user """
    print '\n' # Line break
    print 'Average nearest point distance: ' + str('%1.2f' % nearest)
    dist_tol = int(nearest * 1.75) # Suggested distance tolerance
    print 'Suggested distance tolerance: ' + str(dist_tol)
    print '\n'
    dist_tol = prompt_user_for_number('Enter distance tolerance between points [e.g. ' + str(dist_tol) + ']: ')
    print '\n'
    azimuth_tol = prompt_user_for_number('Enter azimuth tolerance between two segments [e.g. 35]: ')
    print '\n'
    min_line_segments = prompt_user_for_number('Enter minimum number of segments in a line [e.g. 3]: ')
    print '\n'
    return [dist_tol, azimuth_tol, min_line_segments]

def suggested_distance_tolerance(data):
    """ Returns approximate mean closest point distance for distance tolerance """
    k = data.shape[0]
    if k > 500:
        k = 500
    samples = sample(range(data.shape[0]), k)
    total = 0
    for i in samples:
        total += np.sort(np.linalg.norm(data - data[i,:], axis = 1))[1] # Closest distance point
    return total / k

def azimuth_difference(azimuth1, azimuth2):
    """ Returns difference of two azimuths in degrees """
    azimuth_diff = np.absolute(azimuth1 - azimuth2)
    if azimuth_diff > 180:
        azimuth_diff = 360 - azimuth_diff
    return azimuth_diff

def get_azimuth(point1, point2):
    """ Returns heading in degrees clockwise from north """
    dx = float(point2[0] - point1[0])
    dy = float(point2[1] - point1[1])
    if dx > 0: theta = (np.pi*0.5) - np.arctan(dy/dx)
    elif dx < 0: theta = (np.pi*1.5) - np.arctan(dy/dx)
    elif dy > 0: theta = 0
    elif dy < 0: theta = np.pi
    else:
        print "Error: get_azimuth went wrong"
    return np.degrees(theta)

def evaluate_point_score(dist, deviation, params):
    """ Neighbor nodes are scored by their distance and azimuth deviation
    to choose the best neighbor to add to path """
    dist_score = (-1.0/params[0])*(dist - params[0]/2.0)**2 + 50 # parabola
    dev_score = 50.0 * (params[1] - deviation) / params[1]
    return dist_score + dev_score

def grow_path(path, grid, bounds, data, visited, params):
    """ Adds points to path """
    while True:
        azimuth = get_azimuth(data[path[-2]], data[path[-1]])
        curr = path[-1]
        neighbors = get_neighbors(curr, grid, bounds, data, visited, params, path)
        best_next = PriorityQueue() # Keep track of point with highest priority score
        for next in neighbors:
            azimuth_next = get_azimuth(data[path[-1]], data[next])
            dist_next = np.linalg.norm(data[curr] - data[next], axis = 0)
            deviation = azimuth_difference(azimuth, azimuth_next)
            if deviation < params[1]:
                score = evaluate_point_score(dist_next, deviation, params)
                best_next.put((-score, next))
        if best_next.empty():
            break
        else:
            path.append(best_next.get()[1])
    return path

def evaluate_possible_paths(possible_paths):
    """ Returns path with max number of segments"""
    best = 0
    for i, path in enumerate(possible_paths):
        if len(path) > len(possible_paths[best]):
            best = i
    return possible_paths[best]

def valid_neighbors(neighbors, data, curr, visited, dist, path):
    """ Returns neighbors within distance tolerance and not visited or in path """
    result = []
    for idx in neighbors:
        if (np.linalg.norm(data[idx] - data[curr]) < dist):
            if (idx not in visited and idx not in path and idx != curr):
                result.append(idx)
    return result

def get_neighbors(curr, grid, tfun, data, visited, params, path = []):
    """ Returns neighbor nodes in a list.
    Neighbors are defined as being within distance tolerance, not in path,
    and not current node. """
    row, col = tfun(data[curr])
    neighbors = []
    # loop through neighbor cells, grid is padded with empty rows/cols
    for i in range(row - 1, row + 2):
        if i >= 0:
            for j in range(col - 1, col + 2):
                if j >= 0:
                    neighbors += grid[i, j]
    return valid_neighbors(neighbors, data, curr, visited, params[0], path)

def find_best_path(curr, grid, tfun, data, visited, params):
    """ Returns the path with the most segments """
    neighbors = get_neighbors(curr, grid, tfun, data, visited, params)
    possible_paths = []
    for next in neighbors:
        path = [curr, next]
        path = grow_path(path, grid, tfun, data, visited, params)
        path.reverse() # Grow path in opposite direction
        path = grow_path(path, grid, tfun, data, visited, params)
        if len(path) > params[2]: # Check number of segments in path
            possible_paths.append(path)
    if possible_paths:
            return evaluate_possible_paths(possible_paths)
    else:
        return []

def load_queue(num):
    """ Load a random list of integers of length n into a Queue"""
    temp = range(num)
    shuffle(temp)
    queue = Queue()
    for x in temp: queue.put(x)
    return queue

def export_database_with_lines(fname, params, df, lines):
    """ Updates dataframe with lines and write to current directory.
    Params are converted to int for fname string.
     """
    df['Line'] = -1
    df['FID'] = -1
    line_num = 1
    for i, path in enumerate(lines):
        fid = 1
        for idx in path:
            df.at[idx, 'Line'] = line_num
            df.at[idx, 'FID'] = fid
            fid += 1
        line_num += 1
    df = df.loc[df['Line'] != -1]
    df.sort_values(by = ['Line', 'FID'], inplace=True)
    df.drop(['FID'], axis=1, inplace=True)
    out_name = fname[:-4] + '_LINES_' + str(int(params[0])) + '_' + str(int(params[1])) + '_' + str(int(params[2])) + '.csv'
    df.to_csv(out_name, index=False)
    return None

def upscale_func(bounds, dist):
    """ Returns function for upscaled grid indices given lower left location of data """
    def transform(coords):
        loc = (coords - [bounds[0], bounds[2]]) / dist
        return int(loc[1]), int(loc[0])
    return transform

def upscale_data(data, dist, bounds):
    """ Upscales data to grid with cell size equal to distance tolerance """
    ncols = int((bounds[1] - bounds[0]) / dist) + 3 # protects out of bounds
    nrows = int((bounds[3] - bounds[2]) / dist) + 3
    grid = np.asarray([set() for x in range(ncols * nrows)])
    grid = grid.reshape((nrows, ncols))
    tfun = upscale_func(bounds, dist) # upscaled grid indexing
    for idx, coords in enumerate(data):
        grid[tfun(coords)].add(idx)
    return grid, tfun

def find_lines(grid, tfun, data, params):
    """ Returns list of index paths after growing paths from
    each seed point.
    """
    lines = [] # Keep track of line paths
    visited = set() # Keep track of assigned points
    num_points = data.shape[0]
    explore = load_queue(num_points)
    count_progress = 0
    while not explore.empty():
        update_progress(count_progress, num_points)
        curr = explore.get()
        if curr not in visited:
            path = find_best_path(curr, grid, tfun, data, visited, params)
            if path:
                lines.append(path)
                for idx in path:
                    visited.add(idx)
                    count_progress += 1
            else:
                visited.add(curr)
                count_progress += 1
    return lines

def main():
    welcome_message()
    while True:
        fname = prompt_user_for_file('Enter curvature filename [e.g. somedatabase.csv] : ')
        try:
            df = pd.read_csv(fname)
            data = df[['X', 'Y']].values
        except:
            print 'Error: ' + fname + ' is not in the correct format.'
            break
        avg_closest_point = suggested_distance_tolerance(data)
        params = get_inputs_from_user(avg_closest_point)
        bounds = [data[:,0].min(), data[:,0].max(), data[:,1].min(), data[:,1].max()]
        grid, tfun = upscale_data(data, params[0], bounds)
        lines = find_lines(grid, tfun, data, params)
        export_database_with_lines(fname, params, df, lines)
        if not user_end_action():
            break
    end = raw_input("Hit any key to quit: ") # Keeps console window open




if __name__=='__main__':
    main()
