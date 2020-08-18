# Exploiting Julia's parallel processing abilities

Take a look at the random search function below.  Random search is very simple metaheuristic based on a *pure exploitation* strategy.  If the problem is a routing one like in the travelling salesperson problem then we repeatedly shuffle a vector of locations for a given budget (e.g. time or a iteration budget) and then selects the best solution.

```julia
function random_restarts(init_solution, matrix; maxiter=1000)
    best = copy(init_solution)
    neighbour = copy(init_solution)
    best_cost = -tour_cost(best, matrix)
    
    
    iter = 1
    while iter < maxiter
        iter += 1
        neighbour = shuffle(neighbour)
        neighbour_cost = -tour_cost(neighbour, matrix)
        
        if neighbour_cost > best_cost
            best, best_cost = neighbour, neighbour_cost
        end
    end
    
    return -best_cost, best
end 
```
Here is some Julia code to test the non-parallel version of the code. The functions `euclidean_distance` and `trim` can be found in the companion [Jupyter notebook]((https://github.com/julia-healthcare/routing-and-scheduling/blob/master/symmetric_tsp/02_random_restarts.ipynb))

```julia
using CSV
using Random

# download datafile
file = "https://raw.githubusercontent.com/julia-healthcare/routing-and-scheduling/master/symmetric_tsp/data/st70.csv"
download(file, "st70.csv")
st70 = CSV.read("st70.csv")

#set random seed
Random.seed!(42);

n_patients = 20
coords = trim_cities(st70, n_patients);
matrix = euclidean_distance_matrix(coords);
tour = [i for i in 1:size(coords)[1]];

#5m shuffles
@time random_restarts(tour, matrix, maxiter=5_000_000)
```

Out:

```
1.071703 seconds (5.00 M allocations: 1.118 GiB, 2.01% gc time)
(511.90063057877717, [11, 5, 10, 16, 1, 13, 15, 2, 14, 20, 8, 3, 19, 4, 9, 12, 7, 18, 6, 17])
```

## Using Distributed

Modern computers have multiple CPU cores.  It is easy to exploit this in Julia using the `Distributed` package, the `@everywhere` macro and the `remotecall` and `fetch` functions.

```julia
using Distributed # for parallel processes

println(nprocs())
println(nworkers())
```

Out: 
```
1
1
```

We need to set this up to use all of the processes on your computer. For example if you have a machine (like mine) that has 16 cores then you can use them for multiple instances of the algorithm.  Note that you may have more or less CPU cores than me!

```julia
#hold one process back to run the immediate process
addprocs(Sys.CPU_THREADS - 1);
println(nworkers())
```
Out:
```
15
```

## @everywhere

We need to make sure that our Julia code is accessible to each of the processes.  To do this we use the `@everywhere` macro.  We need to do this for `Random` (to allow access to `shuffle`) and `random_restarts`

```julia

@everywhere using Random

@everywhere function random_restarts(init_solution, matrix; maxiter=1000)
    best = copy(init_solution)
    neighbour = copy(init_solution)
    best_cost = -tour_cost(best, matrix)
        
    iter = 1
    while iter < maxiter
        iter += 1
        neighbour = shuffle(neighbour)
        neighbour_cost = -tour_cost(neighbour, matrix)
        
        if neighbour_cost > best_cost
            best, best_cost = neighbour, neighbour_cost
        end
    end
    
    return -best_cost, best
end 

```

## remotecall and fetch

You can think of `remotecall` as scheduling the future execution of code in a specified process.  We pass `remotecall` the function we wish to run, it parameters and the id of the process we wish to use. The function `remotecall` returns a `Future` object that we then fetch when we are ready.

Setting up a single remote call:

```julia

# this will run on process number 2. 
#arguments: function, process_id, function_arguments*
remotecall(random_restarts, 2, tour, matrix, maxiter=5_000_000)
```

out:

```
Future(2, 1, 257, nothing)
```

We can see the `Future` object currently contains nothing. To execute it we call `fetch`

```julia
job = remotecall(random_restarts, 2, tour, matrix, maxiter=5_000_000)
fetch(job)
```
out:
```
(519.0610284426596, [6, 12, 17, 9, 2, 4, 7, 15, 19, 8, 20, 14, 3, 5, 11, 13, 1, 16, 10, 18])
```

## Running multiple instances random_restarts in parallel

We can extend the example above by storing all of our `Future` instances in an Array.  Here we store an amount equal to the number of CPU cores - 1 we have on our computer (15 in my case).  

```julia
n_jobs = Sys.CPU_THREADS - 1
jobs = []

for i in 1:n_jobs
    push!(jobs, remotecall(random_restarts, i+1, tour, matrix, maxiter=5_000_000))
end

```

We will use a comprehension to execute all of the jobs in parallel.  

```julia
@time results = [fetch(job) for job in jobs]
```

out:
```
1.009576 seconds (1.42 k allocations: 53.031 KiB)

15-element Array{Tuple{Float64,Array{Int64,1}},1}:
 (507.6272724833943, [15, 2, 4, 6, 5, 10, 9, 12, 17, 11, 1, 13, 16, 18, 19, 8, 20, 14, 3, 7])
 (510.22548190280025, [17, 10, 12, 5, 11, 1, 16, 13, 15, 19, 2, 4, 3, 20, 8, 7, 14, 6, 18, 9])
 (529.7590382968284, [9, 6, 4, 18, 15, 13, 1, 16, 10, 11, 5, 7, 2, 20, 8, 19, 14, 3, 12, 17])
 (515.7013623203869, [20, 14, 9, 12, 5, 11, 16, 1, 13, 10, 17, 6, 3, 18, 7, 19, 8, 4, 15, 2])
 (516.899529331729, [11, 10, 9, 17, 12, 3, 8, 19, 2, 7, 4, 20, 14, 18, 15, 13, 1, 16, 5, 6])
 (498.1969900233141, [10, 5, 1, 13, 16, 11, 12, 9, 17, 8, 18, 6, 2, 7, 19, 20, 14, 3, 15, 4])
 (517.9637773845324, [14, 20, 2, 18, 10, 11, 5, 16, 1, 13, 12, 6, 17, 9, 4, 7, 15, 19, 3, 8])
 (500.8471103351681, [4, 2, 7, 19, 15, 18, 9, 17, 20, 14, 6, 5, 11, 12, 10, 16, 1, 13, 3, 8])
 (485.87152480349073, [6, 11, 12, 20, 14, 3, 8, 2, 19, 7, 4, 18, 15, 1, 13, 16, 10, 5, 17, 9])
 (484.58061695450897, [11, 5, 16, 1, 13, 2, 19, 15, 18, 7, 3, 20, 14, 8, 4, 9, 17, 6, 12, 10])
 (524.1387244381983, [7, 19, 14, 20, 9, 4, 6, 10, 5, 18, 12, 17, 11, 16, 1, 13, 15, 2, 8, 3])
 (537.3972991333424, [18, 6, 10, 16, 1, 13, 5, 9, 12, 11, 17, 8, 20, 3, 7, 2, 15, 19, 4, 14])
 (520.0023954857832, [20, 14, 3, 9, 17, 10, 11, 12, 7, 15, 2, 18, 1, 13, 16, 5, 6, 4, 19, 8])
 (512.148313768913, [11, 10, 16, 13, 1, 5, 12, 17, 9, 3, 18, 6, 7, 2, 19, 4, 15, 8, 14, 20])
 (516.2583147165655, [19, 7, 18, 2, 4, 15, 13, 16, 11, 10, 5, 1, 12, 17, 6, 3, 14, 9, 20, 8])
```

Our best result was 484 which is better than our initial effort of 511.  To do this we execute 5m x 15 = 75m random shuffles in roughly the same time it took to execute 5m on a single core.


