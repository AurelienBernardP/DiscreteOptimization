# Note that the line 591 has been removed from the original CSV file because
# malformatted.
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

#Defining the MIP model
solver = Model(Ipopt.Optimizer)

#Lower & Upper bounds:   eps <= l <= u <= 1-eps 
@variable(solver,u[1:dimension_d] <= 1.0-EPSI)
@variable(solver,l[1:dimension_d] >= EPSI)
@constraint(solver,u .>= l)

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

#values of u & l should be normalized ?
#increase significantly size of csv and check the speed