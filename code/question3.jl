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
THRESHOLD = 1

#Reading the file
df = CSV.read(CSV_FILE_NAME,DataFrame,header=0,delim=';')

#Remove all datapoints of the set Y, only keep datapoints of set X
df_x = filter(row -> (row[ncol(df)] > THRESHOLD),  df)

#Retrieve the last column to only have input variables in the data frame
df = select!(df, Not(ncol(df)))

#Dimension of the dataset
dimension_d = ncol(df)
nb_X = length(df[:,1])

#Normalizing the input variables
df_array = Array(df)
dt = fit(UnitRangeTransform,dims=1 ,df_array)
norm_df = StatsBase.transform!(dt, df_array)

#Defining the MIP model
solver = Model(Ipopt.Optimizer)

#Lower & Upper bounds eps <= l <= u <= 1-eps 
# 0 <= l < u < 1
@variable(solver,u[1:dimension_d] <= 1.0-eps())
@variable(solver,l[1:dimension_d] >= eps())
@constraint(solver,u .>= l)

#Objective function 
@objective(solver,Max,sum(u[i] - l[i] for i in 1:dimension_d))

# epsi = eps(Float16)
epsi = 0.00000001

#For each input variable, (u-x)(l-x) >= eps
for i in 1:dimension_d
    x_i = norm_df[:,i]
    @constraint(solver,[j = 1:nb_X],(u[i] - x_i[j])*(l[i] - x_i[j]) >= epsi)
end

optimize!(solver) #execute model
@show solver #print
@show objective_value(solver)
@show value.(u)
@show value.(l)

#values of u & l should be normalized ?

# function is_bigger(x,y)
#     if(x > y)
#         return 1
#     end
#     return 0
# end
# @NLconstraint(solver, sum(is_bigger(u[i], x_i[j]) for j in 1:nb_X) + sum(is_bigger(x_i[j],l[i]) for j in 1:nb_X) == nb_X)
