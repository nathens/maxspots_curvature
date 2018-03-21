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
import scipy.spatial.distance as spdist
from random import choice
import os.path
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
    print '@version 3/20/2018 \n'
    print 'This program organizes point data resulting from the usgs_curv4.gx into lines. \n'
    print 'To use this program, export the curvature database as a CSV file \n'
    print '(including header). Coordinates must be projected and labeled "X" and "Y".\n'
    print 'Example header: ID,Z1,Z2,Z3,Z4,Z5,Z6,X,Y,X2,Y2,__X,__Y\n'
    return None

def run_again():
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

def get_inputs_from_user():
    """ Returns inputs from user """
    fname = prompt_user_for_file('Enter curvature filename [e.g. somedatabase.csv] : ')
    print '\n' # Line break
    dist_tol = prompt_user_for_number('Enter distance tolerance between points [e.g. 400]: ')
    print '\n'
    azimuth_tol = prompt_user_for_number('Enter azimuth tolerance between two segments [e.g. 35]: ')
    print '\n'
    min_line_segments = prompt_user_for_number('Enter minimum number of segments in a line [e.g. 3]: ')
    print '\n'
    return fname, [dist_tol, azimuth_tol, min_line_segments]

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

def grow_path(path, data, marked, params):
    """ Adds points to path """
    while (True):
        azimuth = get_azimuth(data[path[-2]], data[path[-1]])
        curr = path[-1]
        neighbors = get_neighbors(curr, data, marked, params, path)
        best_next = []
        for next in neighbors:
            azimuth_next = get_azimuth(data[path[-1]], data[next])
            dist_next = np.linalg.norm(data[curr] - data[next], axis = 0) ## Check this
            deviation = azimuth_difference(azimuth, azimuth_next)
            if deviation < params[1]:
                score = evaluate_point_score(dist_next, deviation, params)
                best_next.append([next, score])
        if not best_next:
            break
        else:
            best_next = sorted(best_next,key=lambda l:l[1], reverse=True)
            path.append(best_next[0][0])
    return path

def evaluate_possible_paths(possible_paths):
    """ Returns path with max number of segments"""
    best = 0
    for i, path in enumerate(possible_paths):
        if len(path) > len(possible_paths[best]):
            best = i
    return possible_paths[best]

def get_neighbors(curr, data, marked, params, path = []):
    """ Returns neighbor nodes in a list.
    Neighbors are defined as being within distance tolerance, not in path,
    and not current node. """
    neighbors = (np.linalg.norm(data - data[curr], axis=1) < params[0]).nonzero()[0]
    return [x for x in neighbors if x not in marked and x not in path and x != curr]

def find_best_path(curr, data, marked, params):
    """ Returns the path with the most segments """
    neighbors = get_neighbors(curr, data, marked, params)
    possible_paths = []
    for next in neighbors:
        path = [curr, next]
        path = grow_path(path, data, marked, params)
        path.reverse()
        path = grow_path(path, data, marked, params)
        if len(path) > params[2]: # Check number of segments in path
            possible_paths.append(path)
    if possible_paths:
            return evaluate_possible_paths(possible_paths)
    else:
        return []

def get_lines(data, params):
    """ Main routine for finding paths to connect points """
    line_map = {} # line number: indices of path
    marked = set() # Keep track of assigned points
    toVisit = range(data.shape[0])
    line_num = 1
    num_points = data.shape[0]
    while (toVisit):
        update_progress(num_points - len(toVisit), num_points)
        curr = choice(toVisit)
        best_path = find_best_path(curr, data, marked, params)
        if best_path:
            line_map[line_num] = best_path
            for x in best_path: marked.add(x)
            [toVisit.remove(x) for x in best_path]
            line_num += 1
        else:
            marked.add(curr)
            toVisit.remove(curr)
    return line_map

def export_dataframe(fname, params, df, line_map):
    """ Updates dataframe with lines and write to current directory.
    Params are converted to int for fname string.
     """
    df['Line'] = -1
    df['FID'] = -1
    for line_num, path in line_map.iteritems():
        fid = 1
        for idx in path:
            df.at[idx, 'Line'] = line_num
            df.at[idx, 'FID'] = fid
            fid += 1
    df = df.loc[df['Line'] != -1]
    df.sort_values(by = ['Line', 'FID'], inplace=True)
    df.drop(['FID'], axis=1, inplace=True)
    out_name = fname[:-4] + '_LINES_' + str(int(params[0])) + '_' + str(int(params[1])) + '_' + str(int(params[2])) + '.csv'
    df.to_csv(out_name, index=False)
    return None

def main():
    welcome_message()
    while True:
        fname, params = get_inputs_from_user()
        df = pd.read_csv(fname)
        data = df[['X', 'Y']].values
        line_map = get_lines(data, params)
        export_dataframe(fname, params, df, line_map)
        if not run_again():
            break
    end = raw_input("Hit any key to quit: ") # Keeps console window open




if __name__=='__main__':
    main()
