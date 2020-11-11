## Nested Resampling {#nested-resampling}

In order to obtain unbiased performance estimates for learners, all parts of the model building (preprocessing and model selection steps) should be included in the resampling, i.e., repeated for every pair of training/test data.
For steps that themselves require resampling like hyperparameter tuning or feature-selection (via the wrapper approach) this results in two nested resampling loops.

<img src="images/nested_resampling.png" width="98%" style="display: block; margin: auto;" />

The graphic above illustrates nested resampling for parameter tuning with 3-fold cross-validation in the outer and 4-fold cross-validation in the inner loop.

In the outer resampling loop, we have three pairs of training/test sets.
On each of these outer training sets parameter tuning is done, thereby executing the inner resampling loop.
This way, we get one set of selected hyperparameters for each outer training set.
Then the learner is fitted on each outer training set using the corresponding selected hyperparameters.
Subsequently, we can evaluate the performance of the learner on the outer test sets.

In [mlr3](https://mlr3.mlr-org.com), you can run nested resampling for free without programming any loops by using the [`mlr3tuning::AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) class.
This works as follows:

1. Generate a wrapped Learner via class [`mlr3tuning::AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) or `mlr3filters::AutoSelect` (not yet implemented).
2. Specify all required settings - see section ["Automating the Tuning"](#autotuner) for help.
3. Call function [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) or [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) with the created [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html).

You can freely combine different inner and outer resampling strategies.

A common setup is prediction and performance evaluation on a fixed outer test set.
This can be achieved by passing the [`Resampling`](https://mlr3.mlr-org.com/reference/Resampling.html) strategy (`rsmp("holdout")`) as the outer resampling instance to either [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) or [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html).

The inner resampling strategy could be a cross-validation one (`rsmp("cv")`) as the sizes of the outer training sets might differ.
Per default, the inner resample description is instantiated once for every outer training set.

Note that nested resampling is computationally expensive.
For this reason we use relatively small search spaces and a low number of resampling iterations in the examples shown below.
In practice, you normally have to increase both.
As this is computationally intensive you might want to have a look at the section on [Parallelization](#parallelization).

### Execution {#nested-resamp-exec}

To optimize hyperparameters or conduct feature selection in a nested resampling you need to create learners using either:

* the [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) class, or
* the `mlr3filters::AutoSelect` class (not yet implemented)

We use the example from section ["Automating the Tuning"](#autotuner) and pipe the resulting learner into a [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) call.


```r
library("mlr3tuning")
task = tsk("iris")
learner = lrn("classif.rpart")
resampling = rsmp("holdout")
measure = msr("classif.ce")
param_set = paradox::ParamSet$new(
  params = list(paradox::ParamDbl$new("cp", lower = 0.001, upper = 0.1)))
terminator = trm("evals", n_evals = 5)
tuner = tnr("grid_search", resolution = 10)

at = AutoTuner$new(learner, resampling, measure = measure,
  param_set, terminator, tuner = tuner)
```

Now construct the [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) call:


```r
resampling_outer = rsmp("cv", folds = 3)
rr = resample(task = task, learner = at, resampling = resampling_outer)
```

```
## INFO  [19:27:26.986] Starting to optimize 1 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [19:27:27.025] Evaluating 1 configuration(s) 
## INFO  [19:27:27.144] Result of batch 1: 
## INFO  [19:27:27.148]     cp classif.ce                                uhash 
## INFO  [19:27:27.148]  0.045    0.06061 9ac34c3c-868c-46df-9a36-4f0c0abac2fb 
## INFO  [19:27:27.150] Evaluating 1 configuration(s) 
## INFO  [19:27:27.351] Result of batch 2: 
## INFO  [19:27:27.353]     cp classif.ce                                uhash 
## INFO  [19:27:27.353]  0.078    0.06061 5d8bdbcf-e2f1-49a9-bcf9-685a3b11d951 
## INFO  [19:27:27.356] Evaluating 1 configuration(s) 
## INFO  [19:27:27.460] Result of batch 3: 
## INFO  [19:27:27.463]     cp classif.ce                                uhash 
## INFO  [19:27:27.463]  0.023    0.06061 c8084740-4986-4a39-ab78-8eb0bd79d755 
## INFO  [19:27:27.466] Evaluating 1 configuration(s) 
## INFO  [19:27:27.570] Result of batch 4: 
## INFO  [19:27:27.572]     cp classif.ce                                uhash 
## INFO  [19:27:27.572]  0.089    0.06061 5461eb05-4fe3-4cba-8d8f-df5a7a0416fb 
## INFO  [19:27:27.574] Evaluating 1 configuration(s) 
## INFO  [19:27:27.665] Result of batch 5: 
## INFO  [19:27:27.667]     cp classif.ce                                uhash 
## INFO  [19:27:27.667]  0.001    0.06061 d60f299a-9f7c-44db-b2d5-7ac15938c1ca 
## INFO  [19:27:27.675] Finished optimizing after 5 evaluation(s) 
## INFO  [19:27:27.676] Result: 
## INFO  [19:27:27.678]     cp learner_param_vals  x_domain classif.ce 
## INFO  [19:27:27.678]  0.045          <list[2]> <list[1]>    0.06061 
## INFO  [19:27:27.726] Starting to optimize 1 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [19:27:27.730] Evaluating 1 configuration(s) 
## INFO  [19:27:27.817] Result of batch 1: 
## INFO  [19:27:27.820]     cp classif.ce                                uhash 
## INFO  [19:27:27.820]  0.067    0.09091 b30f4b61-71a0-4ed8-9ddc-aeac3a1949c0 
## INFO  [19:27:27.822] Evaluating 1 configuration(s) 
## INFO  [19:27:27.911] Result of batch 2: 
## INFO  [19:27:27.913]     cp classif.ce                                uhash 
## INFO  [19:27:27.913]  0.034    0.09091 af2c2488-8c53-4c07-bb82-8688285e2580 
## INFO  [19:27:27.916] Evaluating 1 configuration(s) 
## INFO  [19:27:28.005] Result of batch 3: 
## INFO  [19:27:28.008]     cp classif.ce                                uhash 
## INFO  [19:27:28.008]  0.023    0.09091 f72c68e8-4201-4b73-9440-4360c43b4ab5 
## INFO  [19:27:28.010] Evaluating 1 configuration(s) 
## INFO  [19:27:28.100] Result of batch 4: 
## INFO  [19:27:28.103]     cp classif.ce                                uhash 
## INFO  [19:27:28.103]  0.089    0.09091 dd9f18ee-872f-4064-9d38-23f2815eccd3 
## INFO  [19:27:28.105] Evaluating 1 configuration(s) 
## INFO  [19:27:28.199] Result of batch 5: 
## INFO  [19:27:28.202]     cp classif.ce                                uhash 
## INFO  [19:27:28.202]  0.056    0.09091 929d5e89-2578-4166-bda7-7919a4f377f3 
## INFO  [19:27:28.209] Finished optimizing after 5 evaluation(s) 
## INFO  [19:27:28.210] Result: 
## INFO  [19:27:28.212]     cp learner_param_vals  x_domain classif.ce 
## INFO  [19:27:28.212]  0.067          <list[2]> <list[1]>    0.09091 
## INFO  [19:27:28.262] Starting to optimize 1 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [19:27:28.266] Evaluating 1 configuration(s) 
## INFO  [19:27:28.354] Result of batch 1: 
## INFO  [19:27:28.356]     cp classif.ce                                uhash 
## INFO  [19:27:28.356]  0.023    0.06061 8979dd0d-b6d8-41b3-bfec-7b5763a9ef66 
## INFO  [19:27:28.358] Evaluating 1 configuration(s) 
## INFO  [19:27:28.451] Result of batch 2: 
## INFO  [19:27:28.453]     cp classif.ce                                uhash 
## INFO  [19:27:28.453]  0.012    0.06061 d5057476-e3d6-4ed9-b6d6-d9e8527036de 
## INFO  [19:27:28.456] Evaluating 1 configuration(s) 
## INFO  [19:27:28.549] Result of batch 3: 
## INFO  [19:27:28.551]     cp classif.ce                                uhash 
## INFO  [19:27:28.551]  0.089    0.06061 d9d44fa5-54c8-4a68-9c71-c3bfa350f4cb 
## INFO  [19:27:28.554] Evaluating 1 configuration(s) 
## INFO  [19:27:28.645] Result of batch 4: 
## INFO  [19:27:28.647]     cp classif.ce                                uhash 
## INFO  [19:27:28.647]  0.056    0.06061 2f047967-cbf2-4e44-a274-5f86dd441016 
## INFO  [19:27:28.650] Evaluating 1 configuration(s) 
## INFO  [19:27:28.747] Result of batch 5: 
## INFO  [19:27:28.750]     cp classif.ce                                uhash 
## INFO  [19:27:28.750]  0.034    0.06061 f5c82825-177a-4e57-99e7-09cd96467a61 
## INFO  [19:27:28.756] Finished optimizing after 5 evaluation(s) 
## INFO  [19:27:28.758] Result: 
## INFO  [19:27:28.759]     cp learner_param_vals  x_domain classif.ce 
## INFO  [19:27:28.759]  0.023          <list[2]> <list[1]>    0.06061
```

### Evaluation {#nested-resamp-eval}

With the created [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html) we can now inspect the executed resampling iterations more closely.
See the section on [Resampling](#resampling) for more detailed information about [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html) objects.

For example, we can query the aggregated performance result:


```r
rr$aggregate()
```

```
## classif.ce 
##    0.05333
```

Check for any errors in the folds during execution (if there is not output, warnings or errors recorded, this is an empty `data.table()`:


```r
rr$errors
```

```
## Empty data.table (0 rows and 2 cols): iteration,msg
```

Or take a look at the confusion matrix of the joined predictions:


```r
rr$prediction()$confusion
```

```
##             truth
## response     setosa versicolor virginica
##   setosa         50          0         0
##   versicolor      0         47         5
##   virginica       0          3        45
```
