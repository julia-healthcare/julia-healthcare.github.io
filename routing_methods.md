# General optimisation problems in health
---

# Health care logistics: routing and scheduling

Many healthcare systems manage assets or workforces that they need to deploy geographically.  One example, is a community nursing team.  These are teams of highly skilled nurses that must visit patients in their own home. Another example, is patient transport services where a fleet of non-emergency ambulances pick up patients from their own home and transport them to outpatient appointments in a clinical setting.  These problems are highly complex.  For example, in the community nursing example, patients will have a variety of conditions, treatments may be time dependent (for example, insulin injections), nurses will have mixed skills and staffing will vary over time.

# A single asset

> A jupyter notebook containing the code is available [here](https://github.com/julia-healthcare/routing-and-scheduling/blob/master/symmetric_tsp/01_hill_climbing.ipynb)

For simplicity let's consider a single asset that has to visit patients in their own home and ignore the complex constraints described above.  We will contrive this problem as the famous Travelling Salesperson Problem (TSP). 

## Getting the data into Julia

We will use the st70 dataset from [TSPLib](http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsplib.html).

To load the data into Julia we use the CSV and DataFrame packages.  Note that if you need to install these packages.

```julia
using CSV
using DataFrames
file = "https://raw.githubusercontent.com/julia-healthcare/routing-and-scheduling/master/symmetric_tsp/data/st70.csv"
download(file, "st70.csv")
st70 = CSV.read("st70.csv")
```

## Pre-processing the data

We will trim this dataset to one more managable.  We will choose 8 patients which might be a number similar to what a community nurse visits in a single full day shift.

```julia
# function to trim cities
trim_cities(df, ncities) = Array(df[2:end])[1:ncities, :]

#get the first 8 sets of coordinate pairs
coords = trim_cities(st70, 8);
```

We will then construct a symmetric travel matrix that will we will use to solve the combinatorial optimisation problem.  In our case this will contain euclidean distances.  But in general this might include travel times.
The code loops through the array of coordinate pairs and calculates the euclidean distance for each combination.  Note that the diagonal of the matrix will be 0, as this represents the distance between a patients home and itself.

```julia

#contains the norm function
using LinearAlgebra

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

## Representing a route taken by a nurse

A route or tour taken by a nurse is represented as a vector of length 8. 

```julia
#static creation
tour = [1, 2, 3, 4, 5, 6, 7, 8]

#dynamic (useful for large problems)
n_patients = 8
tour = [i for i in 1:n_patients]
```
Note that we assume in the TSP that a nurse will return to their starting location (index 1).  We don't explicitly represent that in the `Array`, but will account for it when we calculate the quality of a tour.

## Calculating the quality of a route

A simple way to calculate the cost of a tour is to loop through all of the arcs in the tour and lookup their cost
in the travel distance matrix.  

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
    
    #return to the start
    cost += matrix[tour[end], tour[1]]
    
    return cost
end

```
## Basic solution methods

We will solve this problem using a general Hill Climbing algorithm.  The pseudo code is below

```
S = some initial candidate solution 
repeat:
    R = Tweak(Copy(S))

    if Quality(R) > Quality(S):
        S = R
until we have run out of time

return S

```

We will use the `tour_cost` function to represent `Quality`.  The function `Tweak` is a design choice, but in general it is a stochastic change in the tour.  Two classic approaches are:

* A two city exchange: two cities are randomly chosen and swapped.
* A two-opt swap: two cities are randomly selected and the section of the route between them is reversed.

```julia
"""
    simple_tweak(tour)

Randomly select two elements within the 
vector tour and swap them
"""
function simple_tweak(tour)
    sample = rand(1:size(tour)[1], 1, 2)
    tour[sample[1]], tour[sample[2]] = tour[sample[2]], tour[sample[1]]
    return tour
end

"""
    tweak_two_opt(tour)

Randomly select two elements within the 
vector tour and reverse the route between (and including)
them.
"""
function tweak_two_opt(tour)
    sample = rand(1:size(tour)[1], 1, 2)
    tour = reverse(tour, tour[sample[1]], tour[sample[2]])
    return tour
end
```

We are now in a position to code our final algorithm - the hill climber.  This is actually very straightfoward as we have functions that perform the hard work.  It is essentially a loop that tweaks the solution and checks if it is better than we had found before.

One thing to note is that by default a hill climber is **climbing**!  We actually want to descend the objective function to find a minimum.  We don't need to change the logic of the hill climbing algorithm, however.  Instead we 
negate (multiple by -1) the value returned from `tour_cost`.  

One extra thing to remember is that we don't want to search forever!  Here we set a timer (default 2 seconds) that terminates the algorithm.  To get the current time in Julia we use the built in `time()`.

```julia
"""
    hill_climb(initial_solution, matrix; time_limit=2, tweak=simple_tweak)

Iteratively test random candidate solutions in the neigbourhood
of the best solution and adopt the news ones if they are better.
Executes until time_limit is reached (default 2 seconds). 

Returns (best_cost, best_solution)

# Arguments

- intitial_solution::Array: initial tour
- matrix::Array 2x2. costs of travel
- time_limit::int64: maximum run time
- tweak::func(tour::Array): tweak function
"""
function hill_climb(initial_solution, matrix; time_limit=2, tweak=simple_tweak)
        
    best = copy(initial_solution)
    best_cost = -tour_cost(initial_solution, matrix)
    
    start = time()
    while (time() - start) < time_limit
        neighbour = tweak(copy(best))
        neighbour_cost = -tour_cost(neighbour, matrix)
        
        if neighbour_cost > best_cost
            best, best_cost = neighbour, neighbour_cost
        end
        
    end
    return -best_cost, best
end
```


## Bringing it all together

That's it. To run it all in one go use the code below.  Note that the code sets a random seed before each optimisation run in order to reproduce the results.  For fun, try the code with a harder problem.  You can select up to 70 patients to visit.

```julia
using Random
Random.seed!(42);

#setup
coords = trim_cities(st70, 8);
matrix = euclidean_distance_matrix(coords)
tour = [i for i in 1:size(coords)[1]]

#simple swapping of single cities
cost, solution = hill_climb(tour, matrix)
println(cost, solution)

#two-opt swaps
Random.seed!(42);
cost, solution = hill_climb(tour, matrix, tweak=tweak_two_opt)
println(cost, solution)
```
## Exploitation versus exploration 

An even simpler heuristic search is called Random Restarts. Search is conducted by repeatedly shuffling - or randomly restarting - a tour and selecting the best tour at the end of the run.  The code listing below provides an illustrative implementation.  Hill climbing and Random Search are at the opposite end of the search spectrum.  Random Search is pure exploration while Hill Climbing is highly exploitative (greedy) in that it only looks at neighbours of the best solution.

```julia
function random_restarts(init_solution, matrix; time_limit=2.0)
    best = copy(init_solution)
    best_cost = -tour_cost(best, matrix)
    
    start = time()
    iter = 0
    while (time() - start) < time_limit
        iter += 1
        neighbour = shuffle(copy(init_solution))
        neighbour_cost = -tour_cost(neighbour, matrix)
        
        if neighbour_cost > best_cost
            best, best_cost = neighbour, neighbour_cost
        end
    end
    
    return -best_cost, best, iter
end 
```
To try and balance exploration and exploitation we could combine Hill Climbing and Random Search.  In the code below we set a total time limit and periodically hill climb before randomly restarting from a new initial solution.  This helps - to an extent - to avoid getting stuck in what is sometimes called a 'local optima'.

```julia
function hill_climb_with_random_restarts(init_solution, matrix; time_limit=2.0, 
                                         tweak=simple_tweak)

    S = shuffle(init_solution)
    S_cost = -tour_cost(S, matrix)
    best, best_cost = copy(S), S_cost
    start = time()

    #outer loop = overall time budget
    while (time() - start) < time_limit

        #sample a climbing time up to time remaining
        climbing_time = rand(0: time_limit - (time() - start))
        climb_start = time()
        
        #inner loop = stochastic hill climb
        while (time() - climb_start) < climbing_time
            R = tweak(copy(S))
            R_cost = -tour_cost(R, matrix)

            if R_cost > S_cost
                S, S_cost = copy(R), R_cost
            end
        
        end
        
        #update the best if hill climbing found better solution
        if S_cost > best_cost
            best, best_cost = copy(S), S_cost
        end
        
        #random restart!
        S = shuffle(init_solution)
        S_cost -tour_cost(S, matrix)
    end
    
    return -best_cost, best
    
end
```
Let try these methods and do a simple comparison:

```julia
Random.seed!(42);

coords = trim_cities(st70, 20);
matrix = euclidean_distance_matrix(coords);
tour = [i for i in 1:size(coords)[1]];

# tweak with 2 opt.
@time result1 = hill_climb_with_random_restarts(tour, matrix, time_limit=2.0
                                                tweak=tweak_two_opt)

#reset random seed
Random.seed!(42);

#extra time
@time result2 = hill_climb_with_random_restarts(tour, matrix, time_limit=5.0,
                                                tweak=tweak_two_opt)

println(result1)
println(result2)
```

And the results!

```
2.000009 seconds (19.72 M allocations: 3.621 GiB, 5.49% gc time)
5.000005 seconds (73.04 M allocations: 13.152 GiB, 7.87% gc time)
(396.7043800775278, [6, 5, 10, 11, 12, 17, 9, 20, 14, 3, 8, 7, 2, 19, 15, 13, 1, 16, 4, 18])
(362.3478303531909, [14, 20, 8, 19, 15, 13, 1, 16, 5, 10, 11, 12, 9, 17, 6, 18, 4, 2, 7, 3])
```



