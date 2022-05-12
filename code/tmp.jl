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
eps = 0.00000001


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
@variable(model, u[1:dimension])
@variable(model, l[1:dimension])

# Add objective to the solver
@objective(model, Max, sum(u[i] - l[i] for i in 1:dimension))

# Add constraints to the solver
@constraint(model, [i = 1:rows, j = 1:dimension], (l[j] - input[i,j])*(u[j] - input[i,j]) >= eps)
@constraint(model, [i = 1:dimension], minimum(minimum.(input[1:rows, i])) + eps <= l[i])
@constraint(model, [i = 1:dimension], maximum(maximum.(input[1:rows, i])) - eps >= u[i])
@constraint(model, [i = 1:dimension], l[i] <= u[i])

# Solve the optimization problem
optimize!(model)

# Check why solver stopped
# @show termination_status(model)

# @show objective_value(model)
# @show value.(u)
@show value.(l)