# First Improvement Local Search

Solves the Symmetric TSP using First Improvement Local Search.

This is a variation on stochastic hill climbing where instead we search iteratively and take the first improvement found.

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
* localsearch - take the first improvement found.


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
    local_search(init_solution, matrix; time_limit=2.0, tweak!=simple_tweak!)

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
function local_search(init_solution, matrix; time_limit=2.0, tweak=simple_tweak!)
        
    best = copy(init_solution)
    best_cost = -tour_cost(init_solution, matrix)
    n_cities = size(init_solution)[1]
    
    #candidate solution
    candidate = copy(init_solution)
    
    start = time()
    improvement = true
    
    while improvement && (time() - start) < time_limit
        improvement = false
        
        for city_i in 1:n_cities
            for city_j in city_i+1:n_cities
                tweak(candidate, city_i, city_j)
                cost = -tour_cost(candidate, matrix)
                
                if cost > best_cost
                    best, best_cost = candidate, cost
                    improvement = true
                else
                    #reverse swap as no improvement
                    tweak(candidate, city_i, city_j)
                end
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
@time results1  = local_search(shuffle(tour), matrix)

@time results2 = local_search(shuffle(tour), matrix, tweak=tweak_two_opt!)

println(results1)
println(results2)
```

      0.000048 seconds (4 allocations: 752 bytes)
      0.000047 seconds (4 allocations: 752 bytes)
    (420.6518699576129, [3, 14, 20, 9, 17, 12, 11, 4, 2, 7, 18, 6, 5, 10, 16, 1, 13, 15, 19, 8])
    (367.9963120916785, [5, 10, 11, 12, 9, 17, 6, 18, 4, 3, 14, 20, 8, 19, 7, 2, 15, 13, 1, 16])



```julia

```