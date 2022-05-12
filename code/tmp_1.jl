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

fileName = "DataProjetExport.csv"
threshold = 1


# Read CSV file & put it in data
data = CSV.read(fileName, DataFrame, header=0, delim=';')

# Remove data points of set Y
data = filter(row -> (row[ncol(data)] > threshold),  data)

# Remove the last column which is the output & put output in output variable
output = data[!,ncol(data)]
data = select!(data, Not(ncol(data)))

# Retrieve the dimension D of the data set
dimension = ncol(data)
rows = nrow(data)

# Normalize the data set
data_array = Array{Float64}(undef, rows, dimension)
for i in 1:rows
    for j in 1:dimension
        data_array[i,j] = data[i,j]
    end
end
tmp = fit(UnitRangeTransform, dims=1 ,data_array)
input = StatsBase.transform(tmp, data_array)

# Create the solver
model = Model(Ipopt.Optimizer)

# Add variables to the solver
@variable(model, u[1:dimension] <= 1.0- eps(Float32))
@variable(model, l[1:dimension] >= eps(Float32))
@constraint(model,u .>= l)

# Add objective to the solver
@objective(model, Max, sum(u[i] - l[i] for i in 1:dimension))

#For each input variable, (u-x)(l-x) >= eps
for i in 1:dimension
    x_i = input[:,i]
    @constraint(model,[j = 1:rows],(u[i] - x_i[j])*(l[i] - x_i[j]) >= eps(Float32))
end
# Solve the optimization problem
optimize!(model)

# Check why solver stopped
# @show termination_status(model)

# @show objective_value(model)
# @show value.(u)
@show value.(l)