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

def select_inferior(previous_cost,current_cost,temperature):

    threshold = np.exp((previous_cost - current_cost)/temperature)
    if(np.random.uniform() > threshold):
        return True

    return False

def cost_function(lower,upper):

    cube_len = 0
    for i in range(len(lower)):
        cube_len += (upper[i] - lower[i])

    return cost_function

def find_upper_val(sorted_dims,lower_index,dimension):
    return sorted_dims[dimension,lower_index+1,1]

def simulated_annealing_solver(dataframe, initial_T):
    nb_dimesions = len(dataframe.columns)
    nb_elements = len(dataframe.index)
    sorted_dims = []

    for i in range(nb_dimesions):
        dimension = dataframe[i]
        
        sorted_dims.append(sorted(enumerate(dimension), key=lambda j: j[1]))
        #each row of sorted_dims is a list of nb_elements long with structure [(element nb previous to sort), (sorted attribute val)]
    lower_prev = np.zeros(nb_dimesions)
    upper_prev = np.zeros(nb_dimesions)

    lower_curr = np.zeros(nb_dimesions)
    upper_curr = np.zeros(nb_dimesions)

    for i in reversed(range(1,initial_T + nb_dimesions)): # range 1 (to not divide by 0) to initial_T + nb_dims.(To at least run through every dimension once)

        current_dim = i % nb_dimesions
        random_element_pos = np.random.randint(nb_elements)

        lower_curr[current_dim] = sorted_dims[current_dim,random_element_pos,1]
        upper_curr[current_dim] = sorted_dims[current_dim,random_element_pos+1,1]

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
        
    return 
    # uses tge

#Reading the file
dataframe = pd.read_csv(CSV_FILE_NAME,delimiter=';',header=None, skiprows=[590], dtype=float)

#Remove all datapoints of the set Y
datapoints_Y = (dataframe.iloc[:,-1:] < THRESHOLD).values
dataframe = dataframe.drop([i for i, x in enumerate(datapoints_Y) if x])

#Retrieve the last column to only have input variables in the data frame
output_variable = dataframe.iloc[:,-1:]
dataframe = dataframe.iloc[: , :-1]

#Number of input variables
nb_dimesions = len(dataframe.columns)

#Normalizing the input variables
scaler = preprocessing.MinMaxScaler()
fitted_scaler = scaler.fit_transform(dataframe)
normalized_dataframe = pd.DataFrame(fitted_scaler)
print(normalized_dataframe)
#dataframe order is preserved

simulated_annealing_solver(normalized_dataframe, 100)