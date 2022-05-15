# Note that the line 591 has been removed from the original CSV file because
# malformatted.
using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("StatsBase")
Pkg.add("JuMP")
using JuMP
using CSV
using DataFrames
using StatsBase

CSV_FILE_NAME = "DataProjetExport.csv"
THRESHOLD = 1.0
EPSI = 0.0000001 # = eps(Float16,32,64)

#Reading the file
df = CSV.read(CSV_FILE_NAME,DataFrame,header=0,delim=';')

#Retrieve the last column to only have input variables in the data frame
output_variable = df[:,ncol(df)]
df = select!(df, Not(ncol(df)))

#Normalizing the input variables
df_array = Array(df)
dt = fit(UnitRangeTransform,dims=1 ,df_array)
norm_df = StatsBase.transform!(dt, df_array)

#Remove the set Y from dataframe, only keep datapoints of set X
for i=length(output_variable):-1:1
    if(output_variable[i] <= THRESHOLD)
        global norm_df = norm_df[1:end .!=i,:]
    end
end

#Dimension of the remaining dataset
dimension_d = length(norm_df[1,:])
nb_data = length(norm_df[:,1])
nb_intervals = nb_data+1

#For each dimension/col, sort the vector, get the intervals between each datapoint
intervals = zeros(dimension_d, nb_intervals)
for i in 1:dimension_d
    sorted_col_df = sort(norm_df[:,i])
    intervals[i,1] = sorted_col_df[1]
    intervals[i,nb_data+1] = 1.0-sorted_col_df[nb_data]
    for j in 2:nb_data
        intervals[i,j+1] = sorted_col_df[j] - sorted_col_df[j-1]
    end
end
#Question: in the article, they unify identical values, should we do the same?


#Question: for occup, can last column ever be filled ?
occupation = zeros(Bool,nb_data,nb_intervals,dimension_d)
for i in 1:nb_data
    datapoint = norm_df[i,:]
    for j in 1:dimension_d
        
    end
end



#Defining the MIP model
solver = Model()

#Lower & Upper binary vectors
@variable(solver,binary=true,upper[1:dimension_d, 1:(nb_data+1)])
@variable(solver,binary=true,lower[1:dimension_d, 1:(nb_data+1)])

#Lower_ij >= upper_ij
@constraint(solver,lower .>= upper)

#For each dimension, Lower_i+1 >= Lower_i
@constraint(solver,[i = 1:dimension_d, j = 1:nb_data], lower[i,j] <= lower[i,j+1])
@constraint(solver,[i = 1:dimension_d, j = 1:nb_data], upper[i,j] <= upper[i,j+1])

#Objective function 
@objective(solver,Max,sum(u[i] - l[i] for i in 1:dimension_d))

#For each input variable, (u-x)(l-x) >= eps
for i in 1:dimension_d
    x_i = norm_df[:,i]
    @constraint(solver,[j = 1:nb_data],(u[i] - x_i[j])*(l[i] - x_i[j]) >= EPSI)
end

optimize!(solver)
@show solver 
@show objective_value(solver)
@show value.(u)
@show value.(l)
