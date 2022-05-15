#   question3.jl	
#   Course:  [MATH0462] - Discrete optimization
#   Title:   Project - Box search for data mining
#   Authors: Kenan Ozdemir      20164038
#            Aurelien Bertrand  20176639
#   Date:    May 2022
#   
#   This file implements a MIP formulation for the problem regarding the third
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

CSV_FILE_NAME = "DataProjetExport.csv"
THRESHOLD = 1.0
EPSI = 0.0000001 # = eps(Float16,32,64)

#Reading the file
data = CSV.read(CSV_FILE_NAME,DataFrame,header=0,delim=';')

#Retrieve the last column to only have input variables in the data frame
output_variable = data[:,ncol(data)]
data = select!(data, Not(ncol(data)))

#Normalizing the input variables
data_array = Array(data)
dt = fit(UnitRangeTransform,dims=1 ,data_array)
norm_data = StatsBase.transform!(dt, data_array)
dimension_d = length(norm_data[1,:])

#Splitting the dataframe into the set X and Y
norm_data_y = zeros(0,dimension_d)
norm_data_x = zeros(0,dimension_d)
for i=length(output_variable):-1:1
    if(output_variable[i] <= THRESHOLD)
        global norm_data_y = vcat(norm_data_y, norm_data[i,:]')
    else
        global norm_data_x = vcat(norm_data_x, norm_data[i,:]')
    end
end

#Dimensions of the datasets
nb_data_x = length(norm_data_x[:,1])
nb_data_y = length(norm_data_y[:,1])

#Defining the MIP model
solver = Model(Ipopt.Optimizer)
contains(l,u,x) = (l <= x <= u) ? 1 : 0
register(solver, :contains, 3, contains; autodiff=true)

#Lower & Upper bounds:   eps <= l <= u <= 1-eps 
@variable(solver,u[1:dimension_d] <= 1.0-EPSI)
@variable(solver,z)
@variable(solver,l[1:dimension_d] >= EPSI)
@constraint(solver,u .>= l)

#Objective function 
# @NLobjective(solver,Max,sum(sum(contains(l[i],u[i],norm_data_y[j,i]) for j in 1:nb_data_y) for i in 1:dimension_d))
@objective(solver,Max,z)

#For each input variable, (u-x)(l-x) >= eps
for i in 1:dimension_d
    x_i = norm_data_x[:,i]
    @constraint(solver,[j = 1:nb_data_x],(u[i] - x_i[j])*(l[i] - x_i[j]) >= EPSI)

    y_i = norm_data_y[:,i]
    @NLconstraint(solver, sum(contains(l[i],u[i],y_i[j]) for j in 1:nb_data_y) == z )
end

optimize!(solver)
@show solver 
@show objective_value(solver)
@show value.(u)
@show value.(l)

#values of u & l should be normalized ?
