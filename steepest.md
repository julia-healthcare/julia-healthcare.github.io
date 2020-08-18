# Steep Ascent Local Search

Solves the Symmetric TSP using Steepest Ascent Local Search

This is a variation on first improvement hill climbing.  Instead of taking the first improvement we search all neighbours and take the best (steepest gradient) path.

# Add packages and imports


```julia
# import Pkg
#Pkg.add("DataFrames")
#Pkg.add("CSV")
```


```julia
using Distributed
using DataFrames
using CSV
using LinearAlgebra
@everywhere using Random
```


```julia
#set up extra processes for parallel runs
println(nprocs())
println(nworkers())
```

    1
    1



```julia
addprocs(Sys.CPU_THREADS - 1);
println(nworkers())
```

    15


# Load data from csv

This is the 70 city problem from TSP lib.


```julia
st70 = CSV.read("data/st70.csv");
```


```julia
head(st70, 10)
```




<table class="data-frame"><thead><tr><th></th><th>city</th><th>x</th><th>y</th></tr><tr><th></th><th>Int64</th><th>Int64</th><th>Int64</th></tr></thead><tbody><p>10 rows × 3 columns</p><tr><th>1</th><td>1</td><td>64</td><td>96</td></tr><tr><th>2</th><td>2</td><td>80</td><td>39</td></tr><tr><th>3</th><td>3</td><td>69</td><td>23</td></tr><tr><th>4</th><td>4</td><td>72</td><td>42</td></tr><tr><th>5</th><td>5</td><td>48</td><td>67</td></tr><tr><th>6</th><td>6</td><td>58</td><td>43</td></tr><tr><th>7</th><td>7</td><td>81</td><td>34</td></tr><tr><th>8</th><td>8</td><td>79</td><td>17</td></tr><tr><th>9</th><td>9</td><td>30</td><td>23</td></tr><tr><th>10</th><td>10</td><td>42</td><td>67</td></tr></tbody></table>



# Functions

* euclidean_distance_matrix - compute cost matrix
* trim_cities - make problem instance smaller than 70 cities!
* tour_cost - compute cost of tour
* tweak! - randomly swap two cities in tour
* steepest_ascent - take the best improvement found.


```julia
"""
    euclidean_distance_matrix(cities)

Compute a matrix of euclidean distances between
city x, y coordinate pairs

# Arguments
- cities::Array: n x 2 matrix of x, y coordinates

"""
function euclidean_distance_matrix(cities)
    nrows = size(cities)[1]
    matrix = zeros(nrows, nrows)
    
    row = 1
    
    for city1 in 1:nrows
        col = 1
        for city2 in 1:nrows
            matrix[row, col] = norm(cities[city1, 1:2]-cities[city2, 1:2])
            col+=1
        end
        row +=1
    end
        
    return matrix
end
```




    euclidean_distance_matrix




```julia
trim_cities(df, ncities) = Array(df[2:end])[1:ncities, :]
```




    trim_cities (generic function with 1 method)




```julia
"""
    tour_cost(tour, matrix)

Compute the travel cost of tour using
the cost matrix
"""
function tour_cost(tour, matrix)
    cost = 0.0
    for i in 1:size(tour)[1] - 1
        cost += matrix[tour[i], tour[i+1]]
    end
    
    cost += matrix[tour[end], tour[1]]
    
    return cost
end
```




    tour_cost




```julia
@everywhere begin
"""
    simple_tweak!(tour)

Simple swap of elemements in an array
Note this modifies the input array
This is more efficient than returning a copy.
"""
function simple_tweak!(tour, i, j)
    tour[i], tour[j] = tour[j], tour[i]
end
    
end
```


```julia
@everywhere function tweak_two_opt!(tour, i, j)
    reverse!(tour, i, j)
end
```

# Algorithm


```julia
@everywhere begin
"""
    steepest_ascent(init_solution, matrix; time_limit=2.0, tweak!=simple_tweak!)

First improvement local search

Iteratively test candidate solutions in the neigbourhood
of the best solution and adopt the first improvement found.
Executes until time_limit is reached (default 2 seconds) or no improvements
found.

# Arguments

- intial_solution::Array: initial tour
- matrix::Array 2x2. costs of travel
- time_limit::int64: maximum run time
- tweak::func(tour::Array, i::Int, j::Int): tweak function modifies tour in place
    default = simple_tweak!

# Returns
- Tuple (best_cost, best_solution)

"""
function steepest_ascent(init_solution, matrix; time_limit=2.0, tweak=simple_tweak!)
        
    best = copy(init_solution)
    best_cost = -tour_cost(init_solution, matrix)
    n_cities = size(init_solution)[1]
    
    #candidate solution
    candidate = copy(init_solution)
    
    start = time()
    improvement = true
    best_city_i = 1
    best_city_j = 2
    
    while improvement && (time() - start) < time_limit
        improvement = false
    
        for city_i in 1:n_cities
            for city_j in city_i+1:n_cities
                tweak(candidate, city_i, city_j)
                cost = -tour_cost(candidate, matrix)
                
                if cost > best_cost
                    best_cost = cost
                    best_city_i = city_i
                    best_city_j = city_j
                    improvement = true
                end
                
                #reverse swap and continue search
                tweak(candidate, city_i, city_j)
                            
            #revert to best found
            tweak(candidate, best_city_i, best_city_j)
            
            best_solution = copy(candidate)        
                    
            end
        end
    end
    
    return -best_cost, best
end
   
#end @everywhere
end
```

# Example solution


```julia
Random.seed!(42);

coords = trim_cities(st70, 20);
matrix = euclidean_distance_matrix(coords)
tour = [i for i in 1:size(coords)[1]]

# add solution code
@time results1  = steepest_ascent(shuffle(tour), matrix)

@time results2 = steepest_ascent(shuffle(tour), matrix, tweak=tweak_two_opt!)

println(results1)
println(results2)
```

      0.000107 seconds (764 allocations: 178.859 KiB)
      0.000125 seconds (954 allocations: 223.391 KiB)
    (383.1620241548598, [3, 11, 17, 7, 10, 2, 9, 8, 16, 15, 12, 14, 18, 5, 6, 13, 19, 1, 4, 20])
    (365.4766982830359, [10, 15, 14, 2, 20, 4, 6, 3, 18, 19, 9, 11, 7, 8, 13, 1, 12, 17, 5, 16])

