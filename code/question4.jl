#   question4.jl	
#   Course:  [MATH0462] - Discrete optimization
#   Title:   Project - Box search for data_df mining
#   Authors: Kenan Ozdemir      20164038
#            Aurelien Bertrand  20176639
#   Date:    May 2022
#   
#   This file implements a MIP formulation for the problem regarding the fourth
#   question of the project statement.
using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
using CSV
using DataFrames

# Catching filename and thresold of quality output
if isempty(ARGS) || length(ARGS) != 3
    CSV_FILE_NAME = "data/BasicExample1_q4.csv"
    THRESHOLD = 1.0
    DATA_Y_LIMIT = 100
else
    CSV_FILE_NAME = ARGS[1]
    THRESHOLD = parse(Float64, ARGS[2])
    DATA_Y_LIMIT = parse(Float64, ARGS[3])
end

#Reading the file
println("Reading file")
data_df = CSV.read(CSV_FILE_NAME,DataFrame,header=0,delim=';')

#Retrieve the last column to only have input variables in the data_df frame
output_variable = data_df[:,ncol(data_df)]
data_df = select!(data_df, Not(ncol(data_df)))
data = Array(data_df)
dimension_d = length(data_df[1,:])

#Splitting the dataframe into the set X and Y
println("Finding set X and Y")
data_y = zeros(0,dimension_d)
data_x = zeros(0,dimension_d)
for i=length(output_variable):-1:1
    if(output_variable[i] <= THRESHOLD)
        global data_y = vcat(data_y, data[i,:]')
    else
        global data_x = vcat(data_x, data[i,:]')
    end
end

#Dimensions of the data sets
nb_data_y = length(data_y[:,1])
nb_data_x = length(data_x[:,1])

#Setting the box and origin
println("Setting up the heuristic")
nbIteration = 1
box = []
to_explore = collect(1:nb_data_y)
origin = rand(to_explore)
filter!(e->e != origin,to_explore)
append!(box, origin)
println("Selected origin is datapoint at: ", data_y[origin,:])

#Explore all points from the data set Y
while(!(to_explore == []) && (nbIteration <= DATA_Y_LIMIT))
    if(mod(nbIteration,50) == 0)
        println("Number of iteration:", nbIteration)
        println("Current box: ", box)
        println("Current box size: ", length(box))
    end

    #Among points from set Y, find the closest neigbor to selected origin
    min_distance = Inf
    closest_neighbor = 0
    for i in to_explore
        distance = 0.0
        point = data_y[i,:]
        for j in 1:dimension_d
            distance += abs(data_y[origin,j] - point[j])
        end
        if(distance < min_distance)
            min_distance = distance
            closest_neighbor = i
        end
    end


    #Mark the closest neighbor as explored and add it to the box
    filter!(e->e != closest_neighbor,to_explore)
    append!(box,closest_neighbor)

    #Find min&max at each dimension defining the current box
    box_maximums = fill(-Inf, dimension_d)
    box_minimums = fill(Inf, dimension_d)
    for i in box
        for j in 1:dimension_d
            tmp = data_y[i,j]
            if(tmp < box_minimums[j])
                box_minimums[j] = tmp
            end
            if(tmp > box_maximums[j])
                box_maximums[j] = tmp
            end
        end
    end

    #Check if any points from the set x is part of the box
    for i in 1:nb_data_x
        contains = true
        for j in 1:dimension_d
            if(!(box_minimums[j]<=data_x[i,j]<=box_maximums[j]))
                contains = false
            end
        end
        #Remove closest_neighbor from box
        if(contains)
            filter!(e->e != closest_neighbor,box)
        end
    end
    global nbIteration += 1
end


println("Completed iterations")
println("Number of iteration:", nbIteration)
println("Current box: ", box)
println("Current box size: ", length(box))
println("Data set Y size: ", nb_data_y)
