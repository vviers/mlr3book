---
title: "mlr3 book"
author:
  - Marc Becker
  - Martin Binder
  - Bernd Bischl
  - Michel Lang
  - Florian Pfisterer
  - Nicholas G. Reich
  - Jakob Richter
  - Patrick Schratz
  - Raphael Sonabend
date: "2020-11-11"
documentclass: scrbook
bibliography: book.bib
biblio-style: apalike
link-citations: yes
colorlinks: yes
url: 'https\://mlr3book.mlr-org.com/'
github-repo: mlr-org/mlr3book
always_allow_html: true
cover-image: "block.png"
favicon: "favicon.ico"
apple-touch-icon: "apple-touch-icon.png"
apple-touch-icon-size: 180
---

# Citation Info {-}

To cite this book, please use the following information:

```
@misc{
  title = {mlr3 book},
  author = {Marc Becker and Martin Binder and Bernd Bischl and Michel Lang and Florian Pfisterer and Nicholas G. Reich and Jakob Richter and Patrick Schratz and Raphael Sonabend},
  url = {https://mlr3book.mlr-org.com},
  year = {2020},
  month = {11},
  day = {11},
}
```



# Quickstart {-}

As a 30-second introductory example, we will train a decision tree model on the first 120 rows of iris data set and make predictions on the final 30, measuring the accuracy of the trained model.


```r
library("mlr3")
task = tsk("iris")
learner = lrn("classif.rpart")

# train a model of this learner for a subset of the task
learner$train(task, row_ids = 1:120)
# this is what the decision tree looks like
learner$model
```

```
## n= 120 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 120 70 setosa (0.41667 0.41667 0.16667)  
##   2) Petal.Length< 2.45 50  0 setosa (1.00000 0.00000 0.00000) *
##   3) Petal.Length>=2.45 70 20 versicolor (0.00000 0.71429 0.28571)  
##     6) Petal.Length< 4.95 49  1 versicolor (0.00000 0.97959 0.02041) *
##     7) Petal.Length>=4.95 21  2 virginica (0.00000 0.09524 0.90476) *
```

```r
predictions = learner$predict(task, row_ids = 121:150)
predictions
```

```
## <PredictionClassif> for 30 observations:
##     row_id     truth   response
##        121 virginica  virginica
##        122 virginica versicolor
##        123 virginica  virginica
## ---                            
##        148 virginica  virginica
##        149 virginica  virginica
##        150 virginica  virginica
```

```r
# accuracy of our model on the test set of the final 30 rows
predictions$score(msr("classif.acc"))
```

```
## classif.acc 
##      0.8333
```

More examples can be found in the [mlr3gallery](https://mlr3gallery.mlr-org.com), a collection of use cases and examples.

While learning [mlr3](https://mlr3.mlr-org.com), we highly recommend to print out some of our [cheatsheets](https://cheatsheets.mlr-org.com).
