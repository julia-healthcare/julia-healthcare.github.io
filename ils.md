# Iterated Local Search (ILS)

Iterated Local Search (ILS) is a meta-heuristic designed to overcome the problem of hill-climbing algorithms becoming stuck in local optima (good solutions, that are not the global optimum or best).  ILS runs hill-climbing algorithms multiple times and stochastically climbs (or descends) the hill of local-optima.  ILS has proven to be a highly effective meta-heuristic for the TSP.

We will implement ILS in Julia, enable parallel running of the algorithm and test it on a few different sized problems from [TSPLib](http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsplib.html).  You will see that it performs quite well!  The problems we will test are: berlin52, st70 and ch150.  All of these problem have known optimal solutions.  This means we can take a look at how average performance of our ILS implementation.

> A Jupyter notebook containing the full implementation is available at our [Github repo](https://github.com/julia-healthcare/routing-and-scheduling/blob/master/symmetric_tsp/05_ils.ipynb)

## ILS Psuedo code

The code for ILS is fairly straightforward, but there is a couple of concepts you need to understand.  

* **home** - each iteration of ILS starts from a 'home base'.  This home base is updated as we iterate.  There are different ways to update the home base. For example: an epsilon-greedy strategy; 20% of the time we will use the current local optima and 80% of the time we will choose the local optima only if it is better than the current home.
* **perturbation** - a key step in ILS is to take a moderate to large jump from home before running local search.  We will implement this as a 4-Opt swap.
* **history** - ILS has been shown to benefit from memory.  You can think of this as a Tabu List.  It is a list of previously visited starting points that we are forbidden from using.  The tabu list is a fixed size (a hyper parameter) that forgets older homes after a period of time.  

```julia
function iterated_local_search(init_solution)
    best = copy(init_solution)
    home = copy(init_solution)
    candidate = copy(init_solution)
    history = [home]

    while time_remains

        candidate = local_search(candidate)

        if quality(candidate) > quality(best)
            best = copy(candidate)
        end

        home = update_home(home, candidate)

        candidate, history = perturb(home, history)

    end

    return best
end
```

For an in-depth look a `local_search` see our snippets on [routing and scheduling](https://juliahealthcare.org/routing_methods), [first improvement](https://juliahealthcare.org/first_improv) and [steepest ascent](https://juliahealthcare.org/steepest) 

## Updating the home base

At the extreme's `update_home(home, candidate)` might always return the current local optima (which is favouring exploration) or it might only accept a change if the current candidate is of a higher quality than the current home base (which is favouring exploitation).  There are then many options in between.  For example, epsilon-greedy where the algorithm explores 20% of the time and exploits 80% of the time.  These three examples are coded below:

```julia
#exploration focussed
function random_homebase(home_base, home_cost, candidate, 
                         candidate_cost)
    return candidate, candidate_cost
end

#exploitation focussed
function higher_quality_homebase(home_base, home_cost, candidate, 
                                 candidate_cost)
    if candidate_cost > home_cost
        return candidate, candidate_cost
    else
        return home_base, home_cost
    end
end

#attempting to balance exploration and exploitation
function epsilon_greedy_homebase(home_base, home_cost, candidate, 
                                 candidate_cost)
    epsilon = 0.2
    u = rand(0:1)
    if u > epsilon
        return higher_quality_homebase(home_base, home_cost, candidate, 
                                       candidate_cost)
    else
        return random_homebase(home_base, home_cost, candidate, 
                               candidate_cost)
    end
end
```
## Purtubation

In each iteration of ILS the algorithm needs to perturb it homebase.  In conceptual terms, the idea is that this pertubation 'jumps' the starting solution over the current local optima (or hill) and provides an opportunity to find a new local optima on the other side.  A good approach to handle this in the TSP is a 4-opt tweak known as the 'double bridge' move.  In the TSP, a 4-opt move breaks four links.  We then have four segments that we can recombine.  There's multiple ways to do this recombining and one easy to remember on is called the double bridge.

> Conceptually you can think of a tour in 4 segments: A,B,C and D.  If you tour was [1, 2, 3, 4, 5, 6, 7, 8] then these segments are A = [1, 2]; B = [3, 4]; C = [5, 6] and D = [7, 8].  A double bridge move recombines the tour to ADCB.  To add a bit more exploration the implementation below varies the length of the segments (a bit).

```julia
"""
    double_bridge_tweak(tour)

Perform a 4-opt ("double bridge") move on a tour.  
Note: length of segments is stochastic.

Deterministic example:
If we pass the tour [1, 2, 3, 4, 5, 6, 7, 8]
    
A = [1, 2]
B = [3, 4]
C = [5, 6]
D = [7, 8]
    
original_tour = ABCD
double_bridge = ADCB
    
original_tour = [1, 2, 3, 4, 5, 6, 7, 8]
double_bridge = [1, 2, 7, 8, 5, 6, 3, 4]
    
# Arguments:

- tour: vector representing tour between cities e.g.
        [1, 2, 3, 4, 5, 6, 7, 8]

Returns:
--------
vector. representing the tour after stochastic double bridge swap
"""
function double_bridge_tweak(tour)
    
    n = length(tour)
    max_segment_length = convert(Int, floor(n/3))
    
    end1 = rand(1:max_segment_length) 
    end2 = end1 + rand(1:max_segment_length)
    end3 = end2 + rand(1:max_segment_length) 
        
    a = tour[1:end1]
    b = tour[end1+1:end2]
    c = tour[end2+1:end3]
    d = tour[end3+1:end]
        
    #useful to see how the function is splitting the array
    #println(a, b, c, d)
        
    return vcat(a, d, c, b)

end
```

## Using history to increase the effectiveness of pertubation

Optionally ILS can include a fixed size memory of starting points for `local_search`.  You can think of this memory as a 'tabu list' i.e. a list of tours that we are forbidden from using as an initial solution for `local_search`.  In Julia we will implement this as a capacity constrained `DataStructures.Deque`.  When our memory reaches capacity the oldest tabu tour will be forgotten.  We need some custom code to implement this in Julia, but it is relatively simple:

```julia
"""
push history to Deque and pop old history

"""
function push_history!(tour_to_add, history, max_length)
    if length(history) == max_length
        removed = popfirst!(history)
    end

    push!(history, tour_to_add)
end
```

We will then create a function that calls both the double bridge and push history functions that our ILS code will interact with:

```julia
function tabu_double_bridge_tweak(tour, history, tabu_size)
    # perform stochastic double bridge
    candidate = four_opt_tweak(tour)
    
    #perturb until init solution is not in tabu list
    while candidate in history
        candidate = four_opt_tweak(tour)
    end
    
    #add to memory
    push_history!(candidate, history, tabu_size)
        
    return candidate, history
end
```

What size should the tabu list be?  I'm afraid there are no hard and fast answers.  It is a hyperparameter that needs to be tuned for your specific problem.  You may even find that you don't need to include memory.

## Final ILS code

We now have all of the ingredients for our full ILS algorithm.  The final Julia code listing is below.  Note that the function accepts arguments that will vary its behaviour.  For example, we can vary `local_search`, `local_search_time_limit` and `update_home`.  The function has been setup to work on a maximum number of iterations, but it would be simple to adapt it to work for a given time limit instead.

```julia
function iterated_local_search(init_solution, matrix;
                               update_home=random_homebase, 
                               perturb=tabu_double_bridge_tweak, maxiter=50,
                               local_search=local_search,
                               local_search_tweak=simple_tweak!,
                               local_search_time_limit=2.0,
                               tabu_size=10)

    candidate = copy(init_solution)
    home =  copy(candidate)
    best = copy(candidate)
    history = DataStructures.Deque{Array}()
    
    home_cost = -tour_cost(init_solution, matrix)
    best_cost = home_cost
    
    for i in 1:maxiter
        
        #local search algorithm
        candidate_cost, candidate = local_search(candidate, matrix, 
                                                 tweak=local_search_tweak,
                                                 time_limit=local_search_time_limit)
        #is current iteration local optimum best result found?
        if candidate_cost > best_cost
            best_cost = candidate_cost
            best = copy(candidate)
        end
        
        # update homebase
        home, home_cost = update_home(home, home_cost, candidate, 
                                      candidate_cost)
        
        #take a big step away from homebase
        candidate, history = perturb(home, history, tabu_size)
    end
    
    return -best_cost, best
end
```
## Testing out ILS
The data file and code listing are available in our Github repo.
Note TSPLib's distance matrix is integer values, ours are floats so we expect slight differences in answers.

### Berlin52

TSPLib's optimal solution for Berlin52 is 7542.

```julia
Random.seed!(42);

coords = trim_cities(berlin52, 52);
matrix = euclidean_distance_matrix(coords)
tour = [i for i in 1:52]
iter = 500

@time cost, solution = iterated_local_search(shuffle(tour), matrix, 
                                             local_search_tweak=tweak_two_opt!, 
                                             maxiter=iter);

println("Best tour length: $cost")
```

out:
```
0.205549 seconds (25.86 k allocations: 1.971 MiB, 3.82% gc time)
Best tour length: 7544.365901904087
```

### st70

TSPLib's optimal tour length for st70 is 675

```julia
Random.seed!(42);

coords = trim_cities(st70, 70);
matrix = euclidean_distance_matrix(coords)
tour = [i for i in 1:70]
iter = 2_000

@time cost, solution = iterated_local_search(shuffle(tour), matrix, 
                                             local_search_tweak=tweak_two_opt!, 
                                             maxiter=iter,
                                             update_home=epsilon_greedy_homebase);

println("Best tour length: $cost")
```

out:

```
 1.906342 seconds (108.84 k allocations: 9.561 MiB)
Best tour length: 677.1096092748469
```

### ch150

TSPLib's optimal tour length for ch150 is 6528

```julia
Random.seed!(42);

coords = trim_cities(ch150, 150);
matrix = euclidean_distance_matrix(coords)
tour = [i for i in 1:150]
iter = 5_000
tabu_size = 200

@time cost, solution = iterated_local_search(shuffle(tour), matrix, 
                                             local_search_tweak=tweak_two_opt!, 
                                             maxiter=iter,
                                             update_home=epsilon_greedy_homebase,
                                             tabu_size=200);

println("Best tour length: $cost")
```

out:
```
 51.499881 seconds (3.06 M allocations: 125.292 MiB, 0.03% gc time)
Best tour length: 6613.48777525181
```




