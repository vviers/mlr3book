## Feature Selection / Filtering {#fs}




Often, data sets include a large number of features.
The technique of extracting a subset of relevant features is called "feature selection".

The objective of feature selection is to fit the sparse dependent of a model on a subset of available data features in the most suitable manner.
Feature selection can enhance the interpretability of the model, speed up the learning process and improve the learner performance.
Different approaches exist to identify the relevant features.
Two different approaches are emphasized in the literature:
one is called [Filtering](#fs-filtering) and the other approach is often referred to as feature subset selection or [wrapper methods](#fs-wrapper).

What are the differences [@guyon2003;@chandrashekar2014]?

* **Filtering**:
  An external algorithm computes a rank of the features (e.g. based on the correlation to the response).
  Then, features are subsetted by a certain criteria, e.g. an absolute number or a percentage of the number of variables.
  The selected features will then be used to fit a model (with optional hyperparameters selected by tuning).
  This calculation is usually cheaper than "feature subset selection" in terms of computation time.
  All filters are connected via package [mlr3filters](https://mlr3filters.mlr-org.com).
* **Wrapper Methods**:
  Here, no ranking of features is done.
  Instead, an optimization algorithm selects a subset of the features, evaluates the set by calculating the resampled predictive performance, and then
  proposes a new set of features (or terminates).
  A simple example is the sequential forward selection.
  This method is usually computationally very intensive as a lot of models are fitted.
  Also, strictly speaking, all these models would need to be tuned before the performance is estimated.
  This would require an additional nested level in a CV setting.
  After undertaken all of these steps, the final set of selected features is again fitted (with optional hyperparameters selected by tuning).
  Wrapper methods are implemented in the [mlr3fselect](https://mlr3fselect.mlr-org.com) package.
* **Embedded Methods**:
  Many learners internally select a subset of the features which they find helpful for prediction.
  These subsets can usually be queried, as the following example demonstrates:
  
  ```r
  task = tsk("iris")
  learner = lrn("classif.rpart")
  
  # ensure that the learner selects features
  stopifnot("selected_features" %in% learner$properties)
  
  # fit a simple classification tree
  learner = learner$train(task)
  
  # extract all features used in the classification tree:
  learner$selected_features()
  ```
  
  ```
  ## [1] "Petal.Length" "Petal.Width"
  ```

There are also [Ensemble filters](#fs-ensemble) built upon the idea of stacking single filter methods. These are not yet implemented.


### Filters {#fs-filter}

Filter methods assign an importance value to each feature.
Based on these values the features can be ranked.
Thereafter, we are able to select a feature subset.
There is a list of all implemented filter methods in the [Appendix](#list-filters).

### Calculating filter values {#fs-calc}

Currently, only classification and regression tasks are supported.

The first step it to create a new R object using the class of the desired filter method.
Each object of class `Filter` has a `.$calculate()` method which computes the filter values and ranks them in a descending order.


```r
library("mlr3filters")
filter = FilterJMIM$new()

task = tsk("iris")
filter$calculate(task)

as.data.table(filter)
```

```
##         feature  score
## 1:  Petal.Width 1.0000
## 2: Sepal.Length 0.6667
## 3: Petal.Length 0.3333
## 4:  Sepal.Width 0.0000
```

Some filters support changing specific hyperparameters.
This is similar to setting hyperparameters of a [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) using `.$param_set$values`:


```r
filter_cor = FilterCorrelation$new()
filter_cor$param_set
```

```
## <ParamSet>
##        id    class lower upper
## 1:    use ParamFct    NA    NA
## 2: method ParamFct    NA    NA
##                                                                  levels
## 1: everything,all.obs,complete.obs,na.or.complete,pairwise.complete.obs
## 2:                                             pearson,kendall,spearman
##       default value
## 1: everything      
## 2:    pearson
```

```r
# change parameter 'method'
filter_cor$param_set$values = list(method = "spearman")
filter_cor$param_set
```

```
## <ParamSet>
##        id    class lower upper
## 1:    use ParamFct    NA    NA
## 2: method ParamFct    NA    NA
##                                                                  levels
## 1: everything,all.obs,complete.obs,na.or.complete,pairwise.complete.obs
## 2:                                             pearson,kendall,spearman
##       default    value
## 1: everything         
## 2:    pearson spearman
```

Rather than taking the "long" R6 way to create a filter, there is also a built-in shorthand notation for filter creation:


```r
filter = flt("cmim")
filter
```

```
## <FilterCMIM:cmim>
## Task Types: classif, regr
## Task Properties: -
## Packages: praznik
## Feature types: integer, numeric, factor, ordered
```

### Variable Importance Filters {#fs-var-imp-filters}

All [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) with the property "importance" come with integrated feature selection methods.

You can find a list of all learners with this property in the [Appendix](#fs-filter-embedded-list).

For some learners the desired filter method needs to be set during learner creation.
For example, learner `classif.ranger` (in the package [mlr3learners](https://mlr3learners.mlr-org.com)) comes with multiple integrated methods.
See the help page of [`ranger::ranger`](https://www.rdocumentation.org/packages/ranger/topics/ranger).
To use method "impurity", you need to set the filter method during construction.


```r
library("mlr3learners")
lrn = lrn("classif.ranger", importance = "impurity")
```

Now you can use the [`mlr3filters::FilterImportance`](https://mlr3filters.mlr-org.com/reference/FilterImportance.html) class for algorithm-embedded methods to filter a [`Task`](https://mlr3.mlr-org.com/reference/Task.html).


```r
library("mlr3learners")

task = tsk("iris")
filter = flt("importance", learner = lrn)
filter$calculate(task)
head(as.data.table(filter), 3)
```

```
##         feature  score
## 1:  Petal.Width 43.935
## 2: Petal.Length 42.932
## 3: Sepal.Length  9.937
```

### Ensemble Methods {#fs-ensemble}

Work in progress.

### Wrapper Methods {#fs-wrapper}

Wrapper feature selection is supported via the [mlr3fselect](https://mlr3fselect.mlr-org.com) extension package.
At the heart of [mlr3fselect](https://mlr3fselect.mlr-org.com) are the R6 classes:

* [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html), [`FSelectInstanceMultiCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceMultiCrit.html): These two classes describe the feature selection problem and store the results.
* [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html): This class is the base class for implementations of feature selection algorithms.

### The `FSelectInstance` Classes {#fs-wrapper-optimization}

The following sub-section examines the feature selection on the [`Pima`](https://mlr3.mlr-org.com/reference/mlr_tasks_sonar.html) data set which is used to predict whether or not a patient has diabetes.


```r
task = tsk("pima")
print(task)
```

```
## <TaskClassif:pima> (768 x 9)
## * Target: diabetes
## * Properties: twoclass
## * Features (8):
##   - dbl (8): age, glucose, insulin, mass, pedigree, pregnant, pressure,
##     triceps
```
We use the classification tree from [rpart](https://cran.r-project.org/package=rpart).


```r
learner = lrn("classif.rpart")
```

Next, we need to specify how to evaluate the performance of the feature subsets.
For this, we need to choose a [`resampling strategy`](https://mlr3.mlr-org.com/reference/Resampling.html) and a [`performance measure`](https://mlr3.mlr-org.com/reference/Measure.html).


```r
hout = rsmp("holdout")
measure = msr("classif.ce")
```

Finally, one has to choose the available budget for the feature selection.
This is done by selecting one of the available [`Terminators`](https://bbotk.mlr-org.com/reference/Terminator.html):

* Terminate after a given time ([`TerminatorClockTime`](https://bbotk.mlr-org.com/reference/mlr_terminators_clock_time.html))
* Terminate after a given amount of iterations ([`TerminatorEvals`](https://bbotk.mlr-org.com/reference/mlr_terminators_evals.html))
* Terminate after a specific performance is reached ([`TerminatorPerfReached`](https://bbotk.mlr-org.com/reference/mlr_terminators_perf_reached.html))
* Terminate when feature selection does not improve ([`TerminatorStagnation`](https://bbotk.mlr-org.com/reference/mlr_terminators_stagnation.html))
* A combination of the above in an *ALL* or *ANY* fashion ([`TerminatorCombo`](https://bbotk.mlr-org.com/reference/mlr_terminators_combo.html))

For this short introduction, we specify a budget of 20 evaluations and then put everything together into a [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html):


```r
library("mlr3fselect")

evals20 = trm("evals", n_evals = 20)

instance = FSelectInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = hout,
  measure = measure,
  terminator = evals20
)
instance
```

```
## <FSelectInstanceSingleCrit>
## * State:  Not optimized
## * Objective: <ObjectiveFSelect:classif.rpart_on_pima>
## * Search Space:
## <ParamSet>
##          id    class lower upper      levels        default value
## 1:      age ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 2:  glucose ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 3:  insulin ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 4:     mass ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 5: pedigree ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 6: pregnant ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 7: pressure ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 8:  triceps ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## * Terminator: <TerminatorEvals>
## * Terminated: FALSE
## * Archive:
## <ArchiveFSelect>
## Null data.table (0 rows and 0 cols)
```

To start the feature selection, we still need to select an algorithm which are defined via the [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html) class

### The `FSelector` Class

The following algorithms are currently implemented in [mlr3fselect](https://mlr3fselect.mlr-org.com):

* Random Search ([`FSelectorRandomSearch`](https://mlr3fselect.mlr-org.com/reference/FSelectorRandomSearch.html))
* Exhaustive Search ([`FSelectorExhaustiveSearch`](https://mlr3fselect.mlr-org.com/reference/FSelectorExhaustiveSearch.html))
* Sequential Search ([`FSelectorSequential`](https://mlr3fselect.mlr-org.com/reference/FSelectorSequential.html))
* Recursive Feature Elimination ([`FSelectorRFE`](https://mlr3fselect.mlr-org.com/reference/FSelectorRFE.html))
* Design Points ([`FSelectorDesignPoints`](https://mlr3fselect.mlr-org.com/reference/FSelectorDesignPoints.html))

In this example, we will use a simple random search.


```r
fselector = fs("random_search")
```

### Triggering the Tuning {#wrapper-selection-triggering}

To start the feature selection, we simply pass the [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html) to the `$optimize()` method of the initialized [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html). The algorithm proceeds as follows

1. The [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html) proposes at least one feature subset and may propose multiple subsets to improve parallelization, which can be controlled via the setting `batch_size`).
2. For each feature subset, the given [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) is fitted on the [`Task`](https://mlr3.mlr-org.com/reference/Task.html) using the provided [`Resampling`](https://mlr3.mlr-org.com/reference/Resampling.html).
   All evaluations are stored in the archive of the [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html).
3. The [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) is queried if the budget is exhausted.
   If the budget is not exhausted, restart with 1) until it is.
4. Determine the feature subset with the best observed performance.
5. Store the best feature subset as the result in the instance object.
The best featue subset (`$result_feature_set`) and the corresponding measured performance (`$result_y`) can be accessed from the instance.


```r
fselector$optimize(instance)
```

```
## INFO  [19:27:52.801] Starting to optimize 8 parameter(s) with '<FSelectorRandomSearch>' and '<TerminatorEvals>' 
## INFO  [19:27:52.997] Evaluating 10 configuration(s) 
## INFO  [19:27:55.125] Result of batch 1: 
## INFO  [19:27:55.129]    age glucose insulin  mass pedigree pregnant pressure triceps classif.ce 
## INFO  [19:27:55.129]   TRUE   FALSE    TRUE FALSE     TRUE     TRUE     TRUE   FALSE     0.3398 
## INFO  [19:27:55.129]   TRUE    TRUE    TRUE  TRUE     TRUE     TRUE    FALSE    TRUE     0.2070 
## INFO  [19:27:55.129]   TRUE   FALSE   FALSE  TRUE     TRUE    FALSE    FALSE    TRUE     0.3164 
## INFO  [19:27:55.129]  FALSE   FALSE   FALSE  TRUE    FALSE    FALSE     TRUE   FALSE     0.3750 
## INFO  [19:27:55.129]   TRUE    TRUE   FALSE  TRUE    FALSE     TRUE    FALSE    TRUE     0.2148 
## INFO  [19:27:55.129]  FALSE   FALSE   FALSE  TRUE    FALSE    FALSE    FALSE   FALSE     0.3672 
## INFO  [19:27:55.129]   TRUE    TRUE   FALSE  TRUE    FALSE    FALSE    FALSE   FALSE     0.2344 
## INFO  [19:27:55.129]   TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2070 
## INFO  [19:27:55.129]  FALSE    TRUE   FALSE FALSE    FALSE    FALSE     TRUE   FALSE     0.2500 
## INFO  [19:27:55.129]  FALSE   FALSE   FALSE FALSE     TRUE     TRUE    FALSE    TRUE     0.3438 
## INFO  [19:27:55.129]                                 uhash 
## INFO  [19:27:55.129]  235b128e-e21c-424e-8b27-e82d6cfe5782 
## INFO  [19:27:55.129]  b0c8bcb2-9fb1-4fab-8ce5-53d75c65537f 
## INFO  [19:27:55.129]  e235de0c-a514-46a2-bdc8-96293b0cac9b 
## INFO  [19:27:55.129]  ba01155e-7618-4a47-af45-3a8a26842e33 
## INFO  [19:27:55.129]  2853b10d-e82a-4126-bc61-20de250132f8 
## INFO  [19:27:55.129]  48e613ca-d742-4ea3-aae7-b582ffb6d888 
## INFO  [19:27:55.129]  3f22d0ff-324b-4518-ab8d-efad08da8d51 
## INFO  [19:27:55.129]  b641b032-6d77-407b-a3a9-160289a79563 
## INFO  [19:27:55.129]  479ebc51-e0ac-462a-b76c-46046a1affd5 
## INFO  [19:27:55.129]  696e1d3b-fa50-4ba7-bc01-0e1674573dd5 
## INFO  [19:27:55.132] Evaluating 10 configuration(s) 
## INFO  [19:27:57.055] Result of batch 2: 
## INFO  [19:27:57.058]    age glucose insulin  mass pedigree pregnant pressure triceps classif.ce 
## INFO  [19:27:57.058]   TRUE    TRUE    TRUE  TRUE    FALSE    FALSE     TRUE   FALSE     0.2578 
## INFO  [19:27:57.058]   TRUE    TRUE    TRUE  TRUE    FALSE     TRUE    FALSE    TRUE     0.2266 
## INFO  [19:27:57.058]   TRUE    TRUE    TRUE  TRUE     TRUE    FALSE     TRUE    TRUE     0.2188 
## INFO  [19:27:57.058]   TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2070 
## INFO  [19:27:57.058]  FALSE   FALSE   FALSE FALSE    FALSE    FALSE     TRUE   FALSE     0.3672 
## INFO  [19:27:57.058]  FALSE   FALSE   FALSE FALSE    FALSE    FALSE     TRUE   FALSE     0.3672 
## INFO  [19:27:57.058]   TRUE    TRUE   FALSE FALSE     TRUE    FALSE    FALSE    TRUE     0.2500 
## INFO  [19:27:57.058]   TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2070 
## INFO  [19:27:57.058]   TRUE   FALSE   FALSE FALSE    FALSE    FALSE    FALSE   FALSE     0.3359 
## INFO  [19:27:57.058]   TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2070 
## INFO  [19:27:57.058]                                 uhash 
## INFO  [19:27:57.058]  8ba79b29-f069-40ea-886b-cb64a29cea26 
## INFO  [19:27:57.058]  06c975cd-fbf1-4ffb-be26-673e229dca89 
## INFO  [19:27:57.058]  7fcc94c8-1451-4063-86b4-017e69e5e1a2 
## INFO  [19:27:57.058]  67abf22c-e8f1-400d-882a-b5efeaa2d9ee 
## INFO  [19:27:57.058]  5fd17ec4-5423-467c-82a1-1a502a84c5a1 
## INFO  [19:27:57.058]  88d951e4-46f9-4129-8366-36b97a891f43 
## INFO  [19:27:57.058]  f6a934ea-2af3-4eaf-aee3-1dcb686b7d31 
## INFO  [19:27:57.058]  2965f5d0-7c78-4324-81ee-3454dabc37c2 
## INFO  [19:27:57.058]  7efd8d3a-eea9-41b8-8139-667e8d085cf8 
## INFO  [19:27:57.058]  e8e41a36-87b5-4fe5-814e-ff87075e7412 
## INFO  [19:27:57.066] Finished optimizing after 20 evaluation(s) 
## INFO  [19:27:57.067] Result: 
## INFO  [19:27:57.069]   age glucose insulin mass pedigree pregnant pressure triceps 
## INFO  [19:27:57.069]  TRUE    TRUE    TRUE TRUE     TRUE     TRUE    FALSE    TRUE 
## INFO  [19:27:57.069]                                        features  x_domain classif.ce 
## INFO  [19:27:57.069]  age,glucose,insulin,mass,pedigree,pregnant,... <list[8]>      0.207
```

```
##     age glucose insulin mass pedigree pregnant pressure triceps
## 1: TRUE    TRUE    TRUE TRUE     TRUE     TRUE    FALSE    TRUE
##                                          features  x_domain classif.ce
## 1: age,glucose,insulin,mass,pedigree,pregnant,... <list[8]>      0.207
```

```r
instance$result_feature_set
```

```
## [1] "age"      "glucose"  "insulin"  "mass"     "pedigree" "pregnant" "triceps"
```

```r
instance$result_y
```

```
## classif.ce 
##      0.207
```
One can investigate all resamplings which were undertaken, as they are stored in the archive of the [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html) and can be accessed through `$data()` method:


```r
instance$archive$data()
```

```
##       age glucose insulin  mass pedigree pregnant pressure triceps classif.ce
##  1:  TRUE   FALSE    TRUE FALSE     TRUE     TRUE     TRUE   FALSE     0.3398
##  2:  TRUE    TRUE    TRUE  TRUE     TRUE     TRUE    FALSE    TRUE     0.2070
##  3:  TRUE   FALSE   FALSE  TRUE     TRUE    FALSE    FALSE    TRUE     0.3164
##  4: FALSE   FALSE   FALSE  TRUE    FALSE    FALSE     TRUE   FALSE     0.3750
##  5:  TRUE    TRUE   FALSE  TRUE    FALSE     TRUE    FALSE    TRUE     0.2148
##  6: FALSE   FALSE   FALSE  TRUE    FALSE    FALSE    FALSE   FALSE     0.3672
##  7:  TRUE    TRUE   FALSE  TRUE    FALSE    FALSE    FALSE   FALSE     0.2344
##  8:  TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2070
##  9: FALSE    TRUE   FALSE FALSE    FALSE    FALSE     TRUE   FALSE     0.2500
## 10: FALSE   FALSE   FALSE FALSE     TRUE     TRUE    FALSE    TRUE     0.3438
## 11:  TRUE    TRUE    TRUE  TRUE    FALSE    FALSE     TRUE   FALSE     0.2578
## 12:  TRUE    TRUE    TRUE  TRUE    FALSE     TRUE    FALSE    TRUE     0.2266
## 13:  TRUE    TRUE    TRUE  TRUE     TRUE    FALSE     TRUE    TRUE     0.2188
## 14:  TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2070
## 15: FALSE   FALSE   FALSE FALSE    FALSE    FALSE     TRUE   FALSE     0.3672
## 16: FALSE   FALSE   FALSE FALSE    FALSE    FALSE     TRUE   FALSE     0.3672
## 17:  TRUE    TRUE   FALSE FALSE     TRUE    FALSE    FALSE    TRUE     0.2500
## 18:  TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2070
## 19:  TRUE   FALSE   FALSE FALSE    FALSE    FALSE    FALSE   FALSE     0.3359
## 20:  TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2070
##                                    uhash  x_domain           timestamp batch_nr
##  1: 235b128e-e21c-424e-8b27-e82d6cfe5782 <list[8]> 2020-11-11 19:27:55        1
##  2: b0c8bcb2-9fb1-4fab-8ce5-53d75c65537f <list[8]> 2020-11-11 19:27:55        1
##  3: e235de0c-a514-46a2-bdc8-96293b0cac9b <list[8]> 2020-11-11 19:27:55        1
##  4: ba01155e-7618-4a47-af45-3a8a26842e33 <list[8]> 2020-11-11 19:27:55        1
##  5: 2853b10d-e82a-4126-bc61-20de250132f8 <list[8]> 2020-11-11 19:27:55        1
##  6: 48e613ca-d742-4ea3-aae7-b582ffb6d888 <list[8]> 2020-11-11 19:27:55        1
##  7: 3f22d0ff-324b-4518-ab8d-efad08da8d51 <list[8]> 2020-11-11 19:27:55        1
##  8: b641b032-6d77-407b-a3a9-160289a79563 <list[8]> 2020-11-11 19:27:55        1
##  9: 479ebc51-e0ac-462a-b76c-46046a1affd5 <list[8]> 2020-11-11 19:27:55        1
## 10: 696e1d3b-fa50-4ba7-bc01-0e1674573dd5 <list[8]> 2020-11-11 19:27:55        1
## 11: 8ba79b29-f069-40ea-886b-cb64a29cea26 <list[8]> 2020-11-11 19:27:57        2
## 12: 06c975cd-fbf1-4ffb-be26-673e229dca89 <list[8]> 2020-11-11 19:27:57        2
## 13: 7fcc94c8-1451-4063-86b4-017e69e5e1a2 <list[8]> 2020-11-11 19:27:57        2
## 14: 67abf22c-e8f1-400d-882a-b5efeaa2d9ee <list[8]> 2020-11-11 19:27:57        2
## 15: 5fd17ec4-5423-467c-82a1-1a502a84c5a1 <list[8]> 2020-11-11 19:27:57        2
## 16: 88d951e4-46f9-4129-8366-36b97a891f43 <list[8]> 2020-11-11 19:27:57        2
## 17: f6a934ea-2af3-4eaf-aee3-1dcb686b7d31 <list[8]> 2020-11-11 19:27:57        2
## 18: 2965f5d0-7c78-4324-81ee-3454dabc37c2 <list[8]> 2020-11-11 19:27:57        2
## 19: 7efd8d3a-eea9-41b8-8139-667e8d085cf8 <list[8]> 2020-11-11 19:27:57        2
## 20: e8e41a36-87b5-4fe5-814e-ff87075e7412 <list[8]> 2020-11-11 19:27:57        2
```

The associated resampling iterations can be accessed in the [`BenchmarkResult`](https://mlr3.mlr-org.com/reference/BenchmarkResult.html):


```r
instance$archive$benchmark_result$data
```

```
## <ResultData>
##   Public:
##     as_data_table: function (view = NULL, reassemble_learners = TRUE, convert_predictions = TRUE, 
##     clone: function (deep = FALSE) 
##     combine: function (rdata) 
##     data: list
##     initialize: function (data = NULL) 
##     iterations: function (view = NULL) 
##     learners: function (view = NULL, states = TRUE, reassemble = TRUE) 
##     logs: function (view = NULL, condition) 
##     prediction: function (view = NULL, predict_sets = "test") 
##     predictions: function (view = NULL, predict_sets = "test") 
##     resamplings: function (view = NULL) 
##     sweep: function () 
##     task_type: active binding
##     tasks: function (view = NULL, reassemble = TRUE) 
##     uhashes: function (view = NULL) 
##   Private:
##     deep_clone: function (name, value) 
##     get_view_index: function (view)
```

The `uhash` column links the resampling iterations to the evaluated feature subsets stored in `instance$archive$data()`. This allows e.g. to score the included [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html)s on a different measure.

Now the optimized feature subset can be used to subset the task and fit the model on all observations.


```r
task$select(instance$result_feature_set)
learner$train(task)
```

The trained model can now be used to make a prediction on external data.
Note that predicting on observations present in the `task`,  should be avoided.
The model has seen these observations already during feature selection and therefore results would be statistically biased.
Hence, the resulting performance measure would be over-optimistic.
Instead, to get statistically unbiased performance estimates for the current task, [nested resampling](#nested-resampling) is required.

### Automating the Feature Selection {#autofselect}

The [`AutoFSelector`](https://mlr3fselect.mlr-org.com/reference/AutoFSelector.html) wraps a learner and augments it with an automatic feature selection for a given task.
Because the [`AutoFSelector`](https://mlr3fselect.mlr-org.com/reference/AutoFSelector.html) itself inherits from the [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) base class, it can be used like any other learner.
Analogously to the previous subsection, a new classification tree learner is created.
This classification tree learner automatically starts a feature selection on the given task using an inner resampling (holdout).
We create a terminator which allows 10 evaluations, and and uses a simple random search as feature selection algorithm:


```r
library("paradox")
library("mlr3fselect")

learner = lrn("classif.rpart")
terminator = trm("evals", n_evals = 10)
fselector = fs("random_search")

at = AutoFSelector$new(
  learner = learner,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  terminator = terminator,
  fselector = fselector
)
at
```

```
## <AutoFSelector:classif.rpart.fselector>
## * Model: -
## * Parameters: xval=0
## * Packages: rpart
## * Predict Type: response
## * Feature types: logical, integer, numeric, factor, ordered
## * Properties: importance, missings, multiclass, selected_features,
##   twoclass, weights
```

We can now use the learner like any other learner, calling the `$train()` and `$predict()` method.
This time however, we pass it to [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) to compare the optimized feature subset to the complete feature set.
This way, the [`AutoFSelector`](https://mlr3fselect.mlr-org.com/reference/AutoFSelector.html) will do its resampling for feature selection on the training set of the respective split of the outer resampling.
The learner then undertakes predictions using the test set of the outer resampling.
This yields unbiased performance measures, as the observations in the test set have not been used during feature selection or fitting of the respective learner.
This is called [nested resampling](#nested-resampling).

To compare the optimized feature subset with the complete feature set, we can use [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html):


```r
grid = benchmark_grid(
  task = tsk("pima"),
  learner = list(at, lrn("classif.rpart")),
  resampling = rsmp("cv", folds = 3)
)

# avoid console output from mlrfselect
logger = lgr::get_logger("bbotk")
logger$set_threshold("warn")

bmr = benchmark(grid, store_models = TRUE)
bmr$aggregate(msrs(c("classif.ce", "time_train")))
```

```
##    nr      resample_result task_id              learner_id resampling_id iters
## 1:  1 <ResampleResult[21]>    pima classif.rpart.fselector            cv     3
## 2:  2 <ResampleResult[21]>    pima           classif.rpart            cv     3
##    classif.ce time_train
## 1:     0.2786          0
## 2:     0.2396          0
```

Note that we do not expect any significant differences since we only evaluated a small fraction of the possible feature subsets.
