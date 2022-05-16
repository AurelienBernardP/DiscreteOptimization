#   question1.jl	
#   Course:  [MATH0462] - Discrete optimization
#   Title:   Project - Box search for data mining
#   Authors: Kenan Ozdemir      20164038
#            Aurelien Bertrand  20176639
#   Date:    May 2022
#   
#   This file implements a MIP formulation for the problem regarding the first
#   question of the project statement.

using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("StatsBase")
Pkg.add("JuMP")
Pkg.add("HiGHS")
using JuMP
using CSV
using DataFrames
using StatsBase
using HiGHS

# Catching filename and thresold of quality output
if isempty(ARGS) || length(ARGS) != 2
    CSV_FILE_NAME = "data/BasicExample2.csv"
    THRESHOLD = 1.0
else
    CSV_FILE_NAME = ARGS[1]
    THRESHOLD = parse(Float64, ARGS[2])
end

#Reading the file
println("Reading file")
df = CSV.read(CSV_FILE_NAME,DataFrame,header=0,delim=';')

#Retrieve the last column to only have input variables in the data frame
output_variable = df[:,ncol(df)]
df = select!(df, Not(ncol(df)))

#Normalizing the input variables
println("Normalizing dataframe")
df_array = Array(df)
dt = fit(UnitRangeTransform,dims=1 ,df_array)
norm_df = StatsBase.transform!(dt, df_array)
minimums = zeros(1,ncol(df))
maximums = zeros(1,ncol(df))
for i in 1:ncol(df)
    minimums[i] = minimum(df[:,i])
    maximums[i] = maximum(df[:,i])
end

#Remove the set Y from dataframe, only keep datapoints of set X
for i=length(output_variable):-1:1
    if(output_variable[i] <= THRESHOLD)
        global norm_df = norm_df[1:end .!=i,:]
    end
end

#Dimensions of the remaining dataset
dimension_d = length(norm_df[1,:])
nb_data = length(norm_df[:,1])
nb_intervals = nb_data+1

#Sorting dataframe in each dimension
println("Defining various matrices to implement the solution")
sorted_dimensions = zeros(dimension_d, nb_data)
for i in 1:dimension_d
    sorted_dimensions[i,:] = sort(norm_df[:,i])
end

#Define the intervals between each datapoint in the corresponding sorted dimension
intervals = zeros(dimension_d, nb_intervals)
for i in 1:dimension_d
    intervals[i,1] = sorted_dimensions[i,1]
    intervals[i,nb_data+1] = 1.0-sorted_dimensions[i,nb_data]
    for j in 2:nb_data
        intervals[i,j] = sorted_dimensions[i,j] - sorted_dimensions[i,j-1]
    end
end

# Define for each datapoint in which interval at each dimension it belongs to
occupancy = zeros(Bool,nb_data,dimension_d,nb_intervals)
for i in 1:nb_data
    datapoint = norm_df[i,:]
    for j in 1:dimension_d
        l = 1
        while(l<=nb_data && datapoint[j] > sorted_dimensions[j,l])
            l += 1
        end
        occupancy[i,j,l] = 1
    end
end

#Defining the MIP model
println("Defining the MIP formulation")
solver = Model(HiGHS.Optimizer)

#Lower & Upper binary vectors
@variable(solver,integer=true,0<= upper[1:dimension_d, 1:nb_intervals] <= 1)
@variable(solver,integer=true,0<= lower[1:dimension_d, 1:nb_intervals] <= 1)

#Lower_ij >= upper_ij
@constraint(solver,lower .>= upper)

#For each dimension, Lower_i+1 >= Lower_i
@constraint(solver,[i = 1:dimension_d, j = 1:nb_data], lower[i,j] <= lower[i,j+1])
@constraint(solver,[i = 1:dimension_d, j = 1:nb_data], upper[i,j] <= upper[i,j+1])

# Occupancy constraint
for p in 1:nb_data
    @constraint(solver,sum(sum(occupancy[p,i,j] * (lower[i,j] - upper[i,j]) for j in 1:nb_intervals) for i in 1:dimension_d) <= dimension_d-1)
end

#Objective function 
@objective(solver,Max,sum(sum( intervals[i,j] * (lower[i,j] - upper[i,j]) for j in 1:nb_intervals) for i in 1:dimension_d))
println("Execution of the MIP formulation")
optimize!(solver)
@show solver
print("\n")
println("Upper matrix:")
display(value.(upper))
println("\nLower matrix:")
display(value.(lower))
println("\nBox intervals (lower-upper):")
display(value.(lower-upper))

#Objective values
println("\nNormalized objective function value: ",objective_value(solver))
denormalized(n,dimension) = n*maximums[dimension] - n*minimums[dimension] + minimums[dimension]
println("Denormalized objective function value: ", value(sum(sum(denormalized(intervals[i,j],i) * (lower[i,j] - upper[i,j]) for j in 1:nb_intervals) for i in 1:dimension_d)))

