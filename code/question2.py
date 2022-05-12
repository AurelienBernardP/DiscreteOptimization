#Keep in mind that the given csv file is malformatted on line 591
#The solution right now just skips that line
#Another solution would be something like where malformatted ine have Nan values:
#       csv_file = pd.read_csv(CSV_FILE_NAME,delimiter=';',header=None, thousands=None)
#       csv_file = pd.to_numeric(csv_file, errors='coerce')

from sklearn import preprocessing
import pandas as pd
import numpy as np

CSV_FILE_NAME = "DataProjetExport.csv"
THRESHOLD = 1.
EPS = np.finfo(float).eps

def check_output_is_valid(dataframe, lower, upper):
    nb_dimensions = len(dataframe.columns)
    nb_elements = len(dataframe.index)
    for i in range(nb_dimensions):
        #check if lower and upper are correctly place to not have an element between them
        continue
    return

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

def find_upper_val(sorted_dims,lower_index,dimension):
    return sorted_dims[dimension,lower_index+1,1]

def simulated_annealing_solver(dataframe, initial_T):
    nb_dimensions = len(dataframe.columns)
    nb_elements = len(dataframe.index)
    sorted_dims = []

    for i in range(nb_dimensions):
        dimension = dataframe[i]
        sorted_dims.append(sorted(dimension))
        sorted_dims[i].insert(0, 0)# insert lowest value for each dimension
        sorted_dims[i].append(1.)  # insert highest value for each dimension

        #each row of sorted_dims is a list of nb_elements containing the sorted values for that dimension
    lower_prev = np.zeros(len(sorted_dims[0]))
    upper_prev = np.zeros(len(sorted_dims[0]))

    lower_curr = np.zeros(len(sorted_dims[0]))
    upper_curr = np.zeros(len(sorted_dims[0]))
    prev_cost = 0
    curr_cost = 0
    max_k = initial_T + nb_dimensions
    for i in reversed(range(1,max_k)): # range 1 (to not divide by 0) to initial_T + nb_dims.(To at least run through every dimension once)

        if i % 10000 == 0 :
            print("at iteration " + str(i) + " current cost is : " + str(curr_cost),end='\r')
        current_dim = np.random.randint(len(sorted_dims))
        random_element_pos = np.random.randint(len(sorted_dims[0])-1)

        lower_curr[current_dim] = sorted_dims[current_dim][random_element_pos]
        upper_curr[current_dim] = sorted_dims[current_dim][random_element_pos+1]
        
        prev_cost = cost_function(lower_prev,upper_prev)
        curr_cost = cost_function(lower_curr,upper_curr)

        if curr_cost <= prev_cost:
            # replace new better cost in previous values
            lower_prev[current_dim] = lower_curr[current_dim]
            upper_prev[current_dim] = upper_curr[current_dim]

        elif select_inferior(prev_cost,curr_cost,i):
            # replace by worst box only if lucky
            lower_prev[current_dim] = lower_curr[current_dim]
            upper_prev[current_dim] = upper_curr[current_dim]

        else:
            # revert current values to previous ones
            lower_curr[current_dim] = lower_prev[current_dim]
            upper_curr[current_dim] = upper_prev[current_dim]
    
    print("final max : ", (np.minimum(prev_cost,curr_cost)))
    return lower_curr, upper_curr
    # uses tge

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

L,U = simulated_annealing_solver(normalized_dataframe, 1000000)
print('lower = ', L)
print('upper = ', U)