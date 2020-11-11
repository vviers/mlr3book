## Cost-Sensitive Classification {#cost-sens}

In regular classification the aim is to minimize the misclassification rate and thus all types of misclassification errors are deemed equally severe.
A more general setting is cost-sensitive classification.
Cost sensitive classification does not assume that the costs caused by different kinds of errors are equal.
The objective of cost sensitive classification is to minimize the expected costs.

Imagine you are an analyst for a big credit institution.
Let's also assume that a correct decision of the bank would result in 35% of the profit at the end of a specific period.
A correct decision means that the bank predicts that a customer will pay their bills (hence would obtain a loan), and the customer indeed has good credit.
On the other hand, a wrong decision means that the bank predicts that the customer's credit is in good standing, but the opposite is true.
This would result in a loss of 100% of the given loan.

|                           | Good Customer (truth)       | Bad Customer (truth)       |
| :-----------------------: | :-------------------------: | :------------------------: |
| Good Customer (predicted) | + 0.35                      | - 1.0                      |
| Bad Customer (predicted)  | 0                           | 0                          |


Expressed as costs (instead of profit), we can write down the cost-matrix as follows:


```r
costs = matrix(c(-0.35, 0, 1, 0), nrow = 2)
dimnames(costs) = list(response = c("good", "bad"), truth = c("good", "bad"))
print(costs)
```

```
##         truth
## response  good bad
##     good -0.35   1
##     bad   0.00   0
```
An exemplary data set for such a problem is the [`German Credit`](https://mlr3.mlr-org.com/reference/mlr_tasks_german_credit.html) task:


```r
library("mlr3")
task = tsk("german_credit")
table(task$truth())
```

```
## 
## good  bad 
##  700  300
```

The data has 70% customers who are able to pay back their credit, and 30% bad customers who default on the debt.
A manager, who doesn't have any model, could decide to give either everybody a credit or to give nobody a credit.
The resulting costs for the German credit data are:


```r
# nobody:
(700 * costs[2, 1] + 300 * costs[2, 2]) / 1000
```

```
## [1] 0
```

```r
# everybody
(700 * costs[1, 1] + 300 * costs[1, 2]) / 1000
```

```
## [1] 0.055
```

If the average loan is $20,000, the credit institute would lose more than one million dollar if it would grant everybody a credit:


```r
# average profit * average loan * number of customers
0.055 * 20000 * 1000
```

```
## [1] 1100000
```

Our goal is to find a model which minimizes the costs (and thereby maximizes the expected profit).

### A First Model

For our first model, we choose an ordinary logistic regression (implemented in the add-on package [mlr3learners](https://mlr3learners.mlr-org.com)).
We first create a classification task, then resample the model using a 10-fold cross validation and extract the resulting confusion matrix:


```r
library("mlr3learners")
learner = lrn("classif.log_reg")
rr = resample(task, learner, rsmp("cv"))

confusion = rr$prediction()$confusion
print(confusion)
```

```
##         truth
## response good bad
##     good  609 152
##     bad    91 148
```

To calculate the average costs like above, we can simply multiply the elements of the confusion matrix with the elements of the previously introduced cost matrix, and sum the values of the resulting matrix:


```r
avg_costs = sum(confusion * costs) / 1000
print(avg_costs)
```

```
## [1] -0.06115
```

With an average loan of \$20,000, the logistic regression yields the following costs:


```r
avg_costs * 20000 * 1000
```

```
## [1] -1223000
```

Instead of losing over \$1,000,000, the credit institute now can expect a profit of more than \$1,000,000.

### Cost-sensitive Measure

Our natural next step would be to further improve the modeling step in order to maximize the profit.
For this purpose we first create a cost-sensitive classification measure which calculates the costs based on our cost matrix.
This allows us to conveniently quantify and compare modeling decisions.
Fortunately, there already is a predefined measure [`Measure`](https://mlr3.mlr-org.com/reference/Measure.html) for this purpose: [`MeasureClassifCosts`](https://mlr3.mlr-org.com/reference/mlr_measures_classif.costs.html):


```r
cost_measure = msr("classif.costs", costs = costs)
print(cost_measure)
```

```
## <MeasureClassifCosts:classif.costs>
## * Packages: -
## * Range: [-Inf, Inf]
## * Minimize: TRUE
## * Properties: requires_task
## * Predict type: response
```

If we now call [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) or [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html), the cost-sensitive measures will be evaluated.
We compare the logistic regression to a simple featureless learner and to a random forest from package [ranger](https://cran.r-project.org/package=ranger) :


```r
learners = list(
  lrn("classif.log_reg"),
  lrn("classif.featureless"),
  lrn("classif.ranger")
)
cv3 = rsmp("cv", folds = 3)
bmr = benchmark(benchmark_grid(task, learners, cv3))
bmr$aggregate(cost_measure)
```

```
##    nr      resample_result       task_id          learner_id resampling_id
## 1:  1 <ResampleResult[21]> german_credit     classif.log_reg            cv
## 2:  2 <ResampleResult[21]> german_credit classif.featureless            cv
## 3:  3 <ResampleResult[21]> german_credit      classif.ranger            cv
##    iters classif.costs
## 1:     3      -0.05696
## 2:     3       0.05498
## 3:     3      -0.03822
```

As expected, the featureless learner is performing comparably bad.
The logistic regression and the random forest work equally well.


### Thresholding

Although we now correctly evaluate the models in a cost-sensitive fashion, the models themselves are unaware of the classification costs.
They assume the same costs for both wrong classification decisions (false positives and false negatives).
Some learners natively support cost-sensitive classification (e.g., XXX).
However, we will concentrate on a more generic approach which works for all models which can predict probabilities for class labels: thresholding.

Most learners can calculate the probability $p$ for the positive class.
If $p$ exceeds the threshold $0.5$, they predict the positive class, and the negative class otherwise.

For our binary classification case of the credit data, the we primarily want to minimize the errors where the model predicts "good", but truth is "bad" (i.e., the number of false positives) as this is the more expensive error.
If we now increase the threshold to values $> 0.5$, we reduce the number of false negatives.
Note that we increase the number of false positives simultaneously, or, in other words, we are trading false positives for false negatives.


```r
# fit models with probability prediction
learner = lrn("classif.log_reg", predict_type = "prob")
rr = resample(task, learner, rsmp("cv"))
p = rr$prediction()
print(p)
```

```
## <PredictionClassif> for 1000 observations:
##     row_id truth response prob.good prob.bad
##          1  good     good    0.9621  0.03793
##          8  good     good    0.6373  0.36265
##         12   bad      bad    0.1034  0.89664
## ---                                         
##        994  good      bad    0.3088  0.69122
##        998  good     good    0.9320  0.06804
##       1000  good     good    0.8279  0.17210
```

```r
# helper function to try different threshold values interactively
with_threshold = function(p, th) {
  p$set_threshold(th)
  list(confusion = p$confusion, costs = p$score(measures = cost_measure, task = task))
}

with_threshold(p, 0.5)
```

```
## $confusion
##         truth
## response good bad
##     good  604 145
##     bad    96 155
## 
## $costs
## classif.costs 
##       -0.0664
```

```r
with_threshold(p, 0.75)
```

```
## $confusion
##         truth
## response good bad
##     good  470  80
##     bad   230 220
## 
## $costs
## classif.costs 
##       -0.0845
```

```r
with_threshold(p, 1.0)
```

```
## $confusion
##         truth
## response good bad
##     good    0   1
##     bad   700 299
## 
## $costs
## classif.costs 
##         0.001
```

```r
# TODO: include plot of threshold vs performance
```

Instead of manually trying different threshold values, one uses use [`optimize()`](https://www.rdocumentation.org/packages/stats/topics/optimize) to find a good threshold value w.r.t. our performance measure:


```r
# simple wrapper function which takes a threshold and returns the resulting model performance
# this wrapper is passed to optimize() to find its minimum for thresholds in [0.5, 1]
f = function(th) {
  with_threshold(p, th)$costs
}
best = optimize(f, c(0.5, 1))
print(best)
```

```
## $minimum
## [1] 0.7295
## 
## $objective
## classif.costs 
##      -0.08645
```

```r
# optimized confusion matrix:
with_threshold(p, best$minimum)$confusion
```

```
##         truth
## response good bad
##     good  487  84
##     bad   213 216
```

Note that the function [`optimize()`](https://www.rdocumentation.org/packages/stats/topics/optimize) is intended for unimodal functions and therefore may converge to a local optimum here.
See below for better alternatives to find good threshold values.

### Threshold Tuning

Before we start, we have load all required packages:


```r
library(mlr3)
library(mlr3pipelines)
library(mlr3tuning)
```

### Adjusting thresholds: Two strategies

Currently `mlr3pipelines` offers two main strategies towards adjusting `classification thresholds`.
We can either expose the thresholds as a `hyperparameter` of the Learner by using `PipeOpThreshold`.
This allows us to tune the `thresholds` via an outisde optimizer from `mlr3tuning`.

Alternatively, we can also use `PipeOpTuneThreshold` which automatically tunes the threshold after each learner is fit.

In this blog-post, we'll go through both strategies.

### PipeOpThreshold

`PipeOpThreshold` can be put directly after a `Learner`.

A simple example would be:


```r
gr = lrn("classif.rpart", predict_type = "prob") %>>% po("threshold")
l = GraphLearner$new(gr)
```

Note, that `predict_type` = "prob" is required for `po("threshold")` to have any effect.

The `thresholds` are now exposed as a `hyperparameter` of the `GraphLearner` we created:


```r
l$param_set
```

```
## <ParamSetCollection>
##                               id    class lower upper      levels
##  1:       classif.rpart.minsplit ParamInt     1   Inf            
##  2:      classif.rpart.minbucket ParamInt     1   Inf            
##  3:             classif.rpart.cp ParamDbl     0     1            
##  4:     classif.rpart.maxcompete ParamInt     0   Inf            
##  5:   classif.rpart.maxsurrogate ParamInt     0   Inf            
##  6:       classif.rpart.maxdepth ParamInt     1    30            
##  7:   classif.rpart.usesurrogate ParamInt     0     2            
##  8: classif.rpart.surrogatestyle ParamInt     0     1            
##  9:           classif.rpart.xval ParamInt     0   Inf            
## 10:     classif.rpart.keep_model ParamLgl    NA    NA  TRUE,FALSE
## 11:         threshold.thresholds ParamUty    NA    NA            
##            default value
##  1:             20      
##  2: <NoDefault[3]>      
##  3:           0.01      
##  4:              4      
##  5:              5      
##  6:             30      
##  7:              2      
##  8:              0      
##  9:             10     0
## 10:          FALSE      
## 11: <NoDefault[3]>   0.5
```

We can now tune those thresholds fom the outside as follows:

Before `tuning`, we have to define which hyperparameters we want to tune over.
In this example, we only tune over the `thresholds` parameter of the `threshold` pipeop.
you can easily imagine, that we can also jointly tune over additional hyperparameters, i.e. rpart's `cp` parameter.

As the `Task` we aim to optimize for is a binary task, we can simply specify the threshold param:


```r
library(paradox)
ps = ParamSet$new(list(
  ParamDbl$new("threshold.thresholds", lower = 0, upper = 1)
))
```

We now create a `AutoTuner`, which automatically tunes the supplied learner over the `ParamSet` we supplied above.


```r
at = AutoTuner$new(
  learner = l,
  resampling = rsmp("cv", folds = 3L),
  measure = msr("classif.ce"),
  search_space = ps,
  terminator = trm("evals", n_evals = 5L),
  tuner = TunerRandomSearch$new()
)

at$train(tsk("german_credit"))
```

```
## INFO  [19:30:02.012] Starting to optimize 1 parameter(s) with '<OptimizerRandomSearch>' and '<TerminatorEvals>' 
## INFO  [19:30:02.062] Evaluating 1 configuration(s) 
## INFO  [19:30:02.438] Result of batch 1: 
## INFO  [19:30:02.441]  threshold.thresholds classif.ce                                uhash 
## INFO  [19:30:02.441]                0.1727      0.307 33bb405a-4d8c-492b-8071-2116e12af1d0 
## INFO  [19:30:02.445] Evaluating 1 configuration(s) 
## INFO  [19:30:02.771] Result of batch 2: 
## INFO  [19:30:02.773]  threshold.thresholds classif.ce                                uhash 
## INFO  [19:30:02.773]                0.7816       0.32 3981e67a-42ab-4e60-9a1d-87685c9c613f 
## INFO  [19:30:02.777] Evaluating 1 configuration(s) 
## INFO  [19:30:03.101] Result of batch 3: 
## INFO  [19:30:03.103]  threshold.thresholds classif.ce                                uhash 
## INFO  [19:30:03.103]                0.3378      0.291 e9810c7b-2b56-4851-af80-82a3d4abc76b 
## INFO  [19:30:03.107] Evaluating 1 configuration(s) 
## INFO  [19:30:03.429] Result of batch 4: 
## INFO  [19:30:03.431]  threshold.thresholds classif.ce                                uhash 
## INFO  [19:30:03.431]                0.2091        0.3 7a8f9de3-da97-4823-b27a-68c17a3ea81b 
## INFO  [19:30:03.435] Evaluating 1 configuration(s) 
## INFO  [19:30:03.754] Result of batch 5: 
## INFO  [19:30:03.756]  threshold.thresholds classif.ce                                uhash 
## INFO  [19:30:03.756]               0.07717        0.3 197ff6af-c0a3-4ff9-85f6-b7d2e2f83d6b 
## INFO  [19:30:03.773] Finished optimizing after 5 evaluation(s) 
## INFO  [19:30:03.774] Result: 
## INFO  [19:30:03.776]  threshold.thresholds learner_param_vals  x_domain classif.ce 
## INFO  [19:30:03.776]                0.3378          <list[2]> <list[1]>      0.291
```

Inside the `trafo`, we simply collect all set params into a named vector via `map_dbl` and store it
in the `threshold.thresholds` slot expected by the learner.

Again, we create a `AutoTuner`, which automatically tunes the supplied learner over the `ParamSet` we supplied above.


One drawback of this strategy is, that this requires us to fit a new model for each new threshold setting.
While setting a threshold and computing performance is relatively cheap, fitting the learner is often
more computationally demanding.
A better strategy is therefore often to optimize the thresholds separately after each model fit.


### PipeOpTunethreshold

`PipeOpTuneThreshold` on the other hand works together with `PipeOpLearnerCV`.
It directly optimizes the `cross-validated` predictions made by this `PipeOp`.
This is done in order to avoid over-fitting the threshold tuning.

A simple example would be:


```r
gr = po("learner_cv", lrn("classif.rpart", predict_type = "prob")) %>>% po("tunethreshold")
l2 = GraphLearner$new(gr)
```

Note, that `predict_type` = "prob" is required for `po("tunethreshold")` to work.
Additionally, note that this time no `threshold` parameter is exposed, it is automatically tuned internally.


```r
l2$param_set
```

```
## <ParamSetCollection>
##                                         id    class lower upper      levels
##  1:        classif.rpart.resampling.method ParamFct    NA    NA cv,insample
##  2:         classif.rpart.resampling.folds ParamInt     2   Inf            
##  3: classif.rpart.resampling.keep_response ParamLgl    NA    NA  TRUE,FALSE
##  4:                 classif.rpart.minsplit ParamInt     1   Inf            
##  5:                classif.rpart.minbucket ParamInt     1   Inf            
##  6:                       classif.rpart.cp ParamDbl     0     1            
##  7:               classif.rpart.maxcompete ParamInt     0   Inf            
##  8:             classif.rpart.maxsurrogate ParamInt     0   Inf            
##  9:                 classif.rpart.maxdepth ParamInt     1    30            
## 10:             classif.rpart.usesurrogate ParamInt     0     2            
## 11:           classif.rpart.surrogatestyle ParamInt     0     1            
## 12:                     classif.rpart.xval ParamInt     0   Inf            
## 13:               classif.rpart.keep_model ParamLgl    NA    NA  TRUE,FALSE
## 14:           classif.rpart.affect_columns ParamUty    NA    NA            
## 15:                  tunethreshold.measure ParamUty    NA    NA            
## 16:                tunethreshold.optimizer ParamUty    NA    NA            
## 17:                tunethreshold.log_level ParamUty    NA    NA            
##            default      value
##  1: <NoDefault[3]>         cv
##  2: <NoDefault[3]>          3
##  3: <NoDefault[3]>      FALSE
##  4:             20           
##  5: <NoDefault[3]>           
##  6:           0.01           
##  7:              4           
##  8:              5           
##  9:             30           
## 10:              2           
## 11:              0           
## 12:             10          0
## 13:          FALSE           
## 14:  <Selector[1]>           
## 15: <NoDefault[3]> classif.ce
## 16: <NoDefault[3]>      gensa
## 17:  <function[1]>       warn
```

Note that we can set `rsmp("intask")` as a resampling strategy for "learner_cv" in order to evaluate
predictions on the "training" data. This is generally not advised, as it might lead to over-fitting
on the thresholds but can significantlty reduce runtime.


For more information, see the post on Threshold Tuning on the [mlr3 gallery](https://mlr3gallery.mlr-org.com/).
