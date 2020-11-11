## Parallelization {#parallelization}

Parallelization refers to the process of running multiple jobs in parallel, simultaneously.
This process allows for significant savings in computing power.

[mlr3](https://mlr3.mlr-org.com) uses the [future](https://cran.r-project.org/package=future) backends for parallelization.
Make sure you have installed the required packages [future](https://cran.r-project.org/package=future) and [future.apply](https://cran.r-project.org/package=future.apply):

[mlr3](https://mlr3.mlr-org.com) is capable of parallelizing a variety of different scenarios.
One of the most used cases is to parallelize the [`Resampling`](https://mlr3.mlr-org.com/reference/Resampling.html) iterations.
See [Section Resampling](#resampling) for a detailed introduction to resampling.

In the following section, we will use the _spam_ task and a simple classification tree (`"classif.rpart"`) to showcase parallelization.
We use the [future](https://cran.r-project.org/package=future) package to parallelize the resampling by selecting a backend via the function [`future::plan()`](https://www.rdocumentation.org/packages/future/topics/plan).
We use the [`future::multiprocess`](https://www.rdocumentation.org/packages/future/topics/multiprocess) backend here which uses forks (c.f. [`parallel::mcparallel()`](https://www.rdocumentation.org/packages/parallel/topics/mcparallel)) on UNIX based systems and a [`socket cluster`](https://www.rdocumentation.org/packages/parallel/topics/makePSockCluster) on Windows or if running in [RStudio](https://rstudio.com/):



```r
future::plan("multiprocess")

task = tsk("spam")
learner = lrn("classif.rpart")
resampling = rsmp("subsampling")

time = Sys.time()
resample(task, learner, resampling)
Sys.time() - time
```
By default all CPUs of your machine are used unless you specify argument `workers` in [`future::plan()`](https://www.rdocumentation.org/packages/future/topics/plan).

On most systems you should see a decrease in the reported elapsed time.
On some systems (e.g. Windows), the overhead for parallelization is quite large though.
Therefore, it is advised to only enable parallelization for resamplings where each iteration runs at least 10 seconds.

**Choosing the parallelization level**

If you are transitioning from [mlr](https://cran.r-project.org/package=mlr), you might be used to selecting different parallelization levels, e.g. for resampling, benchmarking or tuning.
In [mlr3](https://mlr3.mlr-org.com) this is no longer required.
All kind of events are rolled out on the same level.
Therefore, there is no need to decide whether you want to parallelize the tuning OR the resampling.

In [mlr3](https://mlr3.mlr-org.com) this is no longer required.
All kind of events are rolled out on the same level - there is no need to decide whether you want to parallelize the tuning OR the resampling.

Just lean back and let the machine do the work :-)

### Nested Resampling Parallelization {#nested-resampling-parallelization}

[Nested resampling](#nested-resampling) results in two nested resampling loops.
We can choose different parallelization backends for the inner and outer resampling loop, respectively.
We just have to pass a list of [future](https://cran.r-project.org/package=future) backends:


```r
# Runs the outer loop in parallel and the inner loop sequentially
future::plan(list("multisession", "sequential"))
# Runs the outer loop sequentially and the inner loop in parallel
future::plan(list("sequential", "multisession"))
```

While nesting real parallelization backends is often unintended and causes unnecessary overhead, it is useful in some distributed computing setups.
It can be achieved with [future](https://cran.r-project.org/package=future) by forcing a fixed number of workers for each loop:


```r
# Runs both loops in parallel
future::plan(list(future::tweak("multisession", workers = 2),
                  future::tweak("multisession", workers = 4)))
```

This example would run on 8 cores (`= 2 * 4`) on the local machine.
The [vignette](https://cran.r-project.org/web/packages/future/vignettes/future-3-topologies.html) of the [future](https://cran.r-project.org/package=future) package gives more insight into nested parallelization.
