# Iterated Local Search (ILS)

Iterated Local Search (ILS) is a meta-heuristic designed to overcome the problem of hill-climbing algorithms becoming stuck in local optima (good solutions, that are not the global optimum or best).  ILS runs hill-climbing algorithms multiple times and stochastically climbs (or descends) the hill of local-optima.  ILS has proven to be a highly effective meta-heuristic for the TSP.

We will implement ILS in Julia, enable parallel running of the algorithm and test it on a few different sized problems from [TSPLib](http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsplib.html).  You will see that it performs quite well!  The problems we will test are: berlin52, st70 and ch150.  All of these problem have known optimal solutions.  This means we can take a look at how average performance of our ILS implementation.

> A Jupyter notebook containing the full implementation is available at our [Github repo](https://github.com/julia-healthcare/routing-and-scheduling/blob/master/symmetric_tsp/05_ils.ipynb)

## ILS Psuedo code

The code for ILS is fairly straightforward, but there is a couple of concepts you need to understand.  

* **home** - each iteration of ILS starts from a 'home base'. 
* **perturbation** - a key step in ILS is to take a moderate to large jump from home before running local search.  
* **history** - ILS has been shown to benefit from memory.  You can think of this as a Tabu List.  It is a list of previously visited home bases that we are forbidden from using again.  The tabu list is a fixed size (a hyper parameter) that forgets older homes after a period of time.  

```
procedure iterated_local_search(init_solution)
    best = copy(init_solution)
    home = copy(init_solution)
    candidate = copy(init_solution)
    history = [home]

    while time remains

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

