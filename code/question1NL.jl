#   question1NL.jl	
#   Course:  [MATH0462] - Discrete optimization
#   Title:   Project - Box search for data mining
#   Authors: Kenan Ozdemir      20164038
#            Aurelien Bernard  20176639
#   Date:    May 2022
#   
#   This file implements a MINLP formulation for the problem regarding the first
#   question of the project statement.

using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("StatsBase")
Pkg.add("JuMP")
Pkg.add("Ipopt")
using JuMP
using Ipopt
using CSV
using DataFrames
using StatsBase

EPSI = 0.000001 #Epsilon value

# Catching filename and thresold of quality output
if isempty(ARGS) || length(ARGS) != 2
    CSV_FILE_NAME = "data/DataProjetExport.csv"
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

#Dimension of the remaining dataset
dimension_d = length(norm_df[1,:])
nb_data = length(norm_df[:,1])

#Defining the MIP model
println("Defining the MIP formulation")
solver = Model(Ipopt.Optimizer)

#Lower & Upper bounds:   eps <= l <= u <= 1-eps 
@variable(solver,u[1:dimension_d] <= 1.0-EPSI)
@variable(solver,l[1:dimension_d] >= EPSI)
@constraint(solver,u .>= l)

#For each input variable, (u-x)(l-x) >= eps
for i in 1:dimension_d
    x_i = norm_df[:,i]
    @constraint(solver,[j = 1:nb_data],(u[i] - x_i[j])*(l[i] - x_i[j]) >= EPSI)
end

#Objective function 
@objective(solver,Max,sum(u[i] - l[i] for i in 1:dimension_d))
println("Execution of the MIP formulation")
optimize!(solver)
@show solver 
print("\n")
println("Upper matrix:")
display(value.(u))
println("\nLower matrix:")
display(value.(l))

#Objective values
println("\nNormalized objective function value: ",objective_value(solver))
denormalized(n,dimension) = n*maximums[dimension] - n*minimums[dimension] + minimums[dimension]
println("Denormalized objective function value: ", value(sum(denormalized(u[i],i) - denormalized(l[i],i) for i in 1:dimension_d)))
