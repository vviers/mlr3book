# Introduction and Overview {#introduction}

The [mlr3](https://mlr3.mlr-org.com) [@mlr3] package and [ecosystem](https://github.com/mlr-org/mlr3/wiki/Extension-Packages) provide a generic, object-oriented, and extensible framework for [classification](#tasks), [regression](#tasks), [survival analysis](#survival), and other machine learning tasks for the R language [@R].
We do not implement any [learners](#learners) ourselves, but provide a unified interface to many existing learners in R.
This unified interface provides functionality to extend and combine existing [learners](#learners), intelligently select and tune the most appropriate technique for a [task](#tasks), and perform large-scale comparisons that enable meta-learning.
Examples of this advanced functionality include [hyperparameter tuning](#tuning), [feature selection](#fs), and [ensemble construction](#fs-ensemble). [Parallelization](#parallelization) of many operations is natively supported.

**Target Audience**

[mlr3](https://mlr3.mlr-org.com) provides a domain-specific language for machine learning in R.
We target both **practitioners** who want to quickly apply machine learning algorithms and **researchers** who want to implement, benchmark, and compare their new methods in a structured environment.
The package is a complete rewrite of an earlier version of [mlr](https://mlr.mlr-org.com) that leverages many years of experience to provide a state-of-the-art system that is easy to use and extend.
It is intended for users who have basic knowledge of machine learning and R and who are interested in complex projects that use advanced functionality as well as one-liners to quickly prototype specific tasks.

**Why a Rewrite?**

[mlr](https://mlr.mlr-org.com) [@mlr] was first released to [CRAN](https://cran.r-project.org) in 2013, with the core design and architecture dating back much further.
Over time, the addition of many features has led to a considerably more complex design that made it harder to build, maintain, and extend than we had hoped for.
With hindsight, we saw that some of the design and architecture choices in [mlr](https://mlr.mlr-org.com) made it difficult to support new features, in particular with respect to pipelines.
Furthermore, the R ecosystem as well as helpful packages such as [data.table](https://cran.r-project.org/package=data.table) have undergone major changes in the meantime.
It would have been nearly impossible to integrate all of these changes into the original design of [mlr](https://mlr.mlr-org.com).
Instead, we decided to start working on a reimplementation in 2018, which resulted in the first release of [mlr3](https://mlr3.mlr-org.com) on CRAN in July 2019.
The new design and the integration of further and newly developed R packages (R6, future, data.table) makes [mlr3](https://mlr3.mlr-org.com) much easier to use, maintain, and more efficient compared to [mlr](https://mlr.mlr-org.com).

**Design Principles**

We follow these general design principles in the [mlr3](https://mlr3.mlr-org.com) package and ecosystem.

* Backend over frontend.
  Most packages of the [mlr3](https://mlr3.mlr-org.com) ecosystem focus on processing and transforming data, applying machine learning algorithms, and computing results.
  We do not provide graphical user interfaces (GUIs); visualizations of data and results are provided in extra packages.
* Embrace [R6](https://cran.r-project.org/package=R6) for a clean, object-oriented design, object state-changes, and reference semantics.
* Embrace [data.table](https://cran.r-project.org/package=data.table) for fast and convenient data frame computations.
* Unify container and result classes as much as possible and provide result data in `data.table`s.
    This considerably simplifies the API and allows easy selection and "split-apply-combine" (aggregation) operations.
    We combine `data.table` and `R6` to place references to non-atomic and compound objects in tables and make heavy use of list columns.
* Defensive programming and type safety.
  All user input is checked with [`checkmate`](https://cran.r-project.org/package=checkmate) [@checkmate].
  Return types are documented, and mechanisms popular in base R which "simplify" the result unpredictably (e.g., `sapply()` or the `drop` argument in `[.data.frame`) are avoided.
* Be light on dependencies.
  One of the main maintenance burdens for [mlr](https://mlr.mlr-org.com) was to keep up with changing learner interfaces and behavior of the many packages it depended on.
  We require far fewer packages in [mlr3](https://mlr3.mlr-org.com) to make installation and maintenance easier.

**Package Ecosystem**

[mlr3](https://mlr3.mlr-org.com) requires the following packages:

  * [backports](https://cran.r-project.org/package=backports):
    Ensures backward compatibility with older R releases. Developed by members of the [mlr3](https://mlr3.mlr-org.com) team.
  * [checkmate](https://cran.r-project.org/package=checkmate):
    Fast argument checks. Developed by members of the [mlr3](https://mlr3.mlr-org.com) team.
  * [mlr3misc](https://cran.r-project.org/package=mlr3misc):
    Miscellaneous functions used in multiple mlr3 [extension packages](https://github.com/mlr-org/mlr3/wiki/Extension-Packages).
    Developed by the [mlr3](https://mlr3.mlr-org.com) team.
  * [mlr3measures](https://cran.r-project.org/package=mlr3measures):
    Performance measures for classification and regression. Developed by members of the [mlr3](https://mlr3.mlr-org.com) team.
  * [paradox](https://cran.r-project.org/package=paradox):
    Descriptions of parameters and parameter sets. Developed by the [mlr3](https://mlr3.mlr-org.com) team.
  * [R6](https://cran.r-project.org/package=R6):
    Reference class objects.
  * [data.table](https://cran.r-project.org/package=data.table):
    Extension of R's `data.frame`.
  * [digest](https://cran.r-project.org/package=digest):
    Hash digests.
  * [uuid](https://cran.r-project.org/package=uuid):
    Unique string identifiers.
  * [lgr](https://cran.r-project.org/package=lgr):
    Logging facility.
  * [mlbench](https://cran.r-project.org/package=mlbench):
    A collection of machine learning data sets.

None of these packages adds any extra recursive dependencies to [mlr3](https://mlr3.mlr-org.com).
Additionally, the following packages are suggested for extra functionality:

* For [parallelization](#parallelization), [mlr3](https://mlr3.mlr-org.com) utilizes the [future](https://cran.r-project.org/package=future) and [future.apply](https://cran.r-project.org/package=future.apply) packages.
* To enable progress bars, use [progressr](https://cran.r-project.org/package=progressr).
* To capture output, warnings, and exceptions, [evaluate](https://cran.r-project.org/package=evaluate) and [callr](https://cran.r-project.org/package=callr) can be used.


While [mlr3](https://mlr3.mlr-org.com) provides the base functionality and some of the most fundamental building blocks for machine learning, the following packages extend [mlr3](https://mlr3.mlr-org.com) with capabilities for preprocessing, pipelining, visualizations, additional learners or additional task types:

<img src="images/mlr3verse.svg" width="98%" style="display: block; margin: auto;" />
A complete list with links to the respective repositories can be found on the [wiki page on extension packages](https://github.com/mlr-org/mlr3/wiki/Extension-Packages).
