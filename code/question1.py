#Keep in mind that the given csv file is malformatted on line 591
#The solution right now just skips that line
#Another solution would be something like where malformatted ine have Nan values:
#       csv_file = pd.read_csv(CSV_FILE_NAME,delimiter=';',header=None, thousands=None)
#       csv_file = pd.to_numeric(csv_file, errors='coerce')

from pulp import *
from sklearn import preprocessing
import pandas as pd

CSV_FILE_NAME = "DataProjetExport.csv"
THRESHOLD = 1
EPS = 0.000000000000001

#Reading the file
dataframe = pd.read_csv(CSV_FILE_NAME,delimiter=';',header=None, skiprows=[590], dtype=float)

#Remove all datapoints of the set Y
datapoints_Y = (dataframe.iloc[:,-1:] < THRESHOLD).values
dataframe = dataframe.drop([i for i, x in enumerate(datapoints_Y) if x])

#Retrieve the last column to only have input variables in the data frame
output_variable = dataframe.iloc[:,-1:]
dataframe = dataframe.iloc[: , :-1]

#Number of input variables
dimension_d = len(dataframe.columns)

#Normalizing the input variables
scaler = preprocessing.MinMaxScaler()
fitted_scaler = scaler.fit_transform(dataframe)
normalized_dataframe = pd.DataFrame(fitted_scaler)
print(normalized_dataframe)
#dataframe order is preserved

# #Defining a MIP problem
lp = LpProblem("Q1_MIP_Problem", LpMaximize)

#Defining 2 dictionaries of input variables keyed by indexes
var_keys = list(range(0,dimension_d))
u = LpVariable.dicts("u", var_keys)
l = LpVariable.dicts("l", var_keys)

#The objective function
lp += (lpSum( u[i] - l[i] for i in var_keys))

# Add the constraints
for i in range(0,dimension_d):
    x_i = normalized_dataframe[i]
    lp += (u[i] >= l[i])
    @constraint(solverNL,l[i] >= (min(x_i)+eps))
    @constraint(solverNL,u[i] <= (max(x_i)-eps))
    for j in range(0,len(x_i)):
        lp += ( (u[i] - x_i[j])*(l[i] - x_i[j]) >= EPS)

# # Solve the LP
status = lp.solve(PULP_CBC_CMD(msg=0))
print("Status:", status)

# #Print solution
for var in lp.variables():
    print(var, "=", value(var))
print("OPT =", value(lp.objective))
