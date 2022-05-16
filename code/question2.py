#   question2.py
#   Course:  [MATH0462] - Discrete optimization
#   Title:   Project - Box search for data mining
#   Authors: Kenan Ozdemir      20164038
#            Aurelien Bernard  20176639
#   Date:    May 2022
#   
#   This file implements a heuristic for the problem regarding the second
#   question of the project statement.

from sklearn import preprocessing
import pandas as pd
import numpy as np

CSV_FILE_NAME = "data/DataProjetExport.csv"
THRESHOLD = 1.
EPS = np.finfo(float).eps

def select_inferior(previous_cost,current_cost,temperature):

    threshold = np.exp((previous_cost - current_cost)/temperature)
    if(np.random.uniform() > threshold):
        return True

    return False

def cost_function(lower,upper):

    cube_len = 0
    for i in range(len(lower)):
        cube_len -= (upper[i] - lower[i])

    return cube_len

def get_occupancy_mat_and_intervals_mat(dataframe):
    nb_dimensions = len(dataframe.columns)
    nb_elements = len(dataframe.index)
    sorted_dims = []
    
    for i in range(nb_dimensions):
        dimension = dataframe.iloc[:,i]
        sorted_dims.append(sorted(enumerate(dimension), key=lambda j: j[1]))
        #sorted_dims : list of Nb_dims elements of structure [(previous element index, value)]      

    intervals = np.zeros((nb_dimensions,nb_elements+1))
    for i in range(nb_dimensions):
        for j in range(nb_elements-1):
            intervals[i,j] = (sorted_dims[i][j+1][1] - sorted_dims[i][j][1])
        
        intervals[i,nb_elements] = (1.0 - sorted_dims[i][len(sorted_dims[i])-1][1])
    
    nb_intervals = (np.shape(intervals))[1]

    occupancy_x = np.zeros((nb_elements,nb_dimensions,nb_intervals),dtype=np.int8)
    for i in range(nb_dimensions):
        for j in range(len(sorted_dims[i])):
            occupancy_x[sorted_dims[i][j][0],i,j] = 1

    return occupancy_x, intervals

def check_colision_aux(box_intervals,occupancy_x):

    nb_dims = len(box_intervals)
    nb_intervals = len(box_intervals[0])
    nb_points = np.shape(occupancy_x)[0]
    for i in range(nb_points):
        nb_collisions = 0

        for j in range(nb_dims):
            for k in range(nb_intervals):
                if box_intervals[j,k] == 1 and (box_intervals[j,k] == occupancy_x[i,j,k]):
                    # there can only be one collision in each dimension
                    nb_collisions +=1
                    break

            if nb_collisions <= j :
                # to be a full collision there needs to be colision in each dimension
                break

        if nb_collisions >= nb_dims :
            #there is a point in the new box if there is a point that colides with nb dims of the box
            return True

    return False
def check_colision(box_intervals,occupancy_x, expanded_dim,expanded_seg):
    #new box 
    new_box = box_intervals.copy()
    new_box[expanded_dim,expanded_seg] = 1

    return check_colision_aux(new_box,occupancy_x)

def expand_box_dim(box_intervals,occupancy_x,dimension_to_expand):
    expand_near = np.random.choice([True,False])
    #if expand near expand closest interval from origin, else expand the furthest interval
    expanded_side = -1

    nb_intervals = np.shape(box_intervals)[1]
    if expand_near and box_intervals[dimension_to_expand,0] == 0:
        #expand near and there is room to expand

        for i in range(nb_intervals):
            #find the nearest limit of the box
            if box_intervals[dimension_to_expand,i] == 1:
                expanded_side = i-1

                if check_colision(box_intervals,occupancy_x, dimension_to_expand, expanded_side):
                    # there is a colision -> can't expand
                    return -1
                return expanded_side

    elif box_intervals[dimension_to_expand,(nb_intervals-1)] == 0:
        # expand furthest limit if there is still space to expand

        for i in range(nb_intervals):
            #find the furthest limit of the box
            if box_intervals[dimension_to_expand,i+1] == 0 and box_intervals[dimension_to_expand,i] == 1:
                expanded_side = i+1
                if check_colision(box_intervals,occupancy_x, dimension_to_expand, expanded_side):
                    # there is a colision -> can't expand
                    return -1
                return expanded_side

    return expanded_side

def total_len_from_box_interval(box_intervals, intervals):
    total_len = 0.0
    for i in range(np.shape(intervals)[0]):
        for j in range(np.shape(intervals)[1]):
            total_len += box_intervals[i,j] * intervals[i,j]

    return total_len

def expand_box_heuristic(dataframe,max_nb_expansions):
    occupancy_x, intervals = get_occupancy_mat_and_intervals_mat(dataframe)

    nb_dims = len(intervals)
    nb_intervals = len(intervals[0])
    
    box_intervals = np.zeros((nb_dims,nb_intervals),dtype = np.int8) 
    # greedy initial box
    for i in range(nb_dims):
        max_interval_distance = 0
        max_interval_index = 0
        for j in range(nb_intervals):
            if intervals[i,j] > max_interval_distance:
                max_interval_distance = intervals[i,j]
                max_interval_index = j
        box_intervals[i,max_interval_index] = 1
    
    first_cell_collision = check_colision_aux(box_intervals,occupancy_x)
    
    if first_cell_collision:
        print('is there collision in first cell ', first_cell_collision)
        coll = True
        while coll:
            intervals_to_set = np.random.randint(nb_intervals, size=nb_dims)
            box_intervals = np.zeros((nb_dims,nb_intervals),dtype = np.int8) 
            for i in range(nb_dims):
                box_intervals[i,intervals_to_set[i]] = 1
            coll = check_colision_aux(box_intervals,occupancy_x)
    print("initial box : ", box_intervals)
    print("initial len : ", total_len_from_box_interval(box_intervals,intervals))

    #start expansion
    expanded = True
    expansion_counter = 0
    while expanded and expansion_counter < max_nb_expansions:
        expanded = False
        print("expanded times = " + str(expansion_counter))
        print("len after expansion : ", total_len_from_box_interval(box_intervals,intervals))
        col = check_colision_aux(box_intervals,occupancy_x)
        if(col):
            print( 'colsiion after expanding ' + str(expansion_counter)  +' times' )
            return box_intervals
        for i in range(nb_dims):
            expanded_segment = expand_box_dim(box_intervals,occupancy_x,i)
            if expanded_segment < 0:
                continue
            else:
                box_intervals[i,expanded_segment] = 1
                expansion_counter += 1
                expanded = True
    
    print("expanded times = " + str(expansion_counter))
    print("len after expansion : ", total_len_from_box_interval(box_intervals,intervals))
    return box_intervals

#################### Begin program


#Reading the file
dataframe = pd.read_csv(CSV_FILE_NAME,delimiter=';',header=None, skiprows=[590], dtype=float)

#Retrieve the last column to only have input variables in the data frame
output_variable = dataframe.iloc[:,-1:]
dataframe = dataframe.iloc[: , :-1]

#Normalizing the input variables
scaler = preprocessing.MinMaxScaler()
fitted_scaler = scaler.fit_transform(dataframe)
normalized_dataframe = pd.DataFrame(fitted_scaler)

#Remove all datapoints of the set Y
for i in range(len(output_variable), 0, -1):
    if(output_variable.iloc[i-1,0] <= THRESHOLD):
        normalized_dataframe = normalized_dataframe.drop(i-1, axis=0)

print(normalized_dataframe)


#dataframe order is preserved
'''
L,U = simple_greedy_approach(normalized_dataframe)
print('lower = ', L)
print('upper = ', U)
'''
box_intervals = expand_box_heuristic(normalized_dataframe,8000)
print("box intervals: ",box_intervals)
np.savetxt('box_intervals.txt', box_intervals) 