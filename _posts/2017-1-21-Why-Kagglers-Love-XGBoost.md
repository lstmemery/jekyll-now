One of the more delightfully named theorems in data science is called "The No Free Lunch Theorem." It states "any two algorithms are equivalent when their performance is averaged across all possible problems."(4) If that's true, why did over half of the winning solutions for the data science competition website [Kaggle](https://www.kaggle.com) in 2015 contain XGBoost?(1) How does XGBoost even work? What does it look like in action?

I intend to answer these questions and intend to do it without asking you to do (too much) math. One thing you will have to know about is decision trees. A beautiful and intuitive introduction to decision trees can be found [here](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/).

## Gradient Boosting

Overfitting is the machine learning equivalent of cramming. If you only memorized the material without drawing out generalizations, you'll flub the final exam. Decision trees are excellent crammers so data scientists have developed special techniques to make sure that they don't overfit.

One method is called boosting. Instead of training one strong learner (the overfitting decision tree) we instead train many weak (underfitting) learners in sequence. The trick is that the current tree is informed about which examples the previous tree got wrong. In order for boosting to work, the learners need to be wrong in different ways. This way, the weaknesses found in some learners can compensated by others. 

A classic implementation is adaptive boosting (AdaBoost), where the misclassified examples of the previous tree are given additional weight. Thus, the current tree has extra incentive to get to previously misclassified examples right.

Gradient boosting is another flavor of boosting. The first tree fits normally, but the second tree tries to find the decisions that would minimize the errors of the of the previous tree. This is difficult because we don't know the mathematical function that describes our data. If we did, we would just use the function! 

Imagine you on top of a hill enveloped in fog. You can't see more than a step in front of you. What be reasonable a reasonable way of finding the fastest way down? One way would be to put your foot out in each direction and feel the steepest way down. After you've done that in every direction, step down the steepest slope. Repeat until you are at the bottom. This is called gradient descent. When this is used to find a tree that minimize the errors of the previous tree it is called a gradient boosted tree algorithm.


<figure>
    <img src='{{ site.baseurl }}/assets/gradient_descent.gif' alt='Gradient Descent' />
    <figcaption>Yu L. Demonstration of the Gradient Descent Algorithm. Vistat [Internet]. 2013 Mar 24 [cited 2017 Jan 21]; Available from: http://vis.supstat.com/2013/03/gradient-descent-algorithm-with-r/</figcaption>
</figure>

## What makes XGBoost EXTREME?

The first proposal for gradient boosted trees was published in 2001.(3) What has XGBoost boost done to improve the algorithm?

One common misconception I've seen is that XGBoost is somehow more accurate than other implementations of the gradient boosted tree algorithm. It's not true! In the original XGBoost paper, the authors compare their implementation to the Sci-Kit Learn implementation and find that it performs about as well.

### XGBoost is not extremely accurate; XGBoost is extremely fast

1. **XGBoost has sparsity-awareness:** Boosted trees work especially well on categorical feature (e.g. Was this person born England?). In many real-world data sets, a categorical feature column will be mostly zeros. When deciding where to split, XGBoost has indices of the non-zero data are and only needs to look at those entries.
2. **XGBoost is parallelizable:** The most time-consuming step for boosted tree algorithms is sorting continuous features (e.g. How far do you drive to work each day?). XGBoost takes advantage of the fact it's sparse data structure allows columns to be sorted independently. This way, the sorting work can be divided up between parallel threads of the CPU. 
3. **XGBoost can make approximate cuts:** In order to find the most effective cut on a continuous feature, gradient boosted trees need to keep all of the data in memory at the same time to sort it. This is not a problem for small datasets but it becomes impossible when you have more data than RAM. XGBoost has to bin these numbers in approximate order instead of sorting them entirely. The authors of the XGBoost paper show that, with enough bins, the you get approximately the same performance as with the exact cut.

## An illustration

Let's take a look at what XGBoost can do. I'll be working with the training data from the Ames Housing Dataset, which can be found [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

To prepare this data, I used Human Analog's code graciously provided via a [Kaggle kernel](https://www.kaggle.com/humananalog/house-prices-advanced-regression-techniques/xgboost-lasso/code).  

This script cleans up the dataset, get's it into a format that xgboost can read and synthesizes new features from the the existing data. For example, he adds a feature that describes whether the house was last renovated the year it was sold.


```python
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor



X = pd.read_csv('../../results/munged_training.csv')
y = pd.read_csv('../../results/munged_labels.csv')

X_train, X_validation, y_train, y_validation = train_test_split(X, y, random_state=1848)
```

    /home/memery/anaconda3/envs/xgboost/lib/python3.5/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
sklearn_boost = GradientBoostingRegressor(random_state=1849)
sklearn_boost.fit(X_train, y_train.values.ravel())
print(1 - sklearn_boost.score(X_train, y_train))
print(1 - sklearn_boost.score(X_validation, y_validation))
%timeit sklearn_boost.fit(X_train, y_train.values.ravel())
```

    0.0314922846261
    0.109576545998
    1 loop, best of 3: 1.23 s per loop



```python
xgb_boost = xgb.XGBRegressor(seed=1850)
xgb_boost.fit(X_train, y_train.values.ravel())
print(1 - xgb_boost.score(X_train, y_train))
print(1 - xgb_boost.score(X_validation, y_validation))
%timeit xgb_boost.fit(X_train, y_train.values.ravel())
```

    0.0380900690956
    0.11111261209
    1 loop, best of 3: 463 ms per loop



```python
ada_boost = AdaBoostRegressor(random_state=1851)
ada_boost.fit(X_train, y_train.values.ravel())
print(1 - ada_boost.score(X_train, y_train))
print(1 - ada_boost.score(X_validation, y_validation))
%timeit ada_boost.fit(X_train, y_train.values.ravel())
```

    0.126232603941
    0.196231101656
    1 loop, best of 3: 756 ms per loop



```python
random_forest = RandomForestRegressor(random_state=1852)
random_forest.fit(X_train, y_train.values.ravel())
print(1 - random_forest.score(X_train, y_train))
print(1 - random_forest.score(X_validation, y_validation))
%timeit random_forest.fit(X_train, y_train.values.ravel())
```

    0.0239502194301
    0.12823390801
    1 loop, best of 3: 391 ms per loop


This is a head-to-head comparison between XGBoost, Gradient Boosted Trees, AdaBoost and Random Forests. XGBoost has a roughly equivalent accuracy score

In this trial XGBoost is about as accurate as the sci-kit learn implementation

#TODO: Write conclusions

## Bibliography

1. Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. arXiv:160302754 [cs]. 2016;785–94. 

2. Mayr A, Binder H, Gefeller O, Schmid M. The Evolution of Boosting Algorithms - From Machine Learning to Statistical Modelling. Methods of Information in Medicine. 2014 Aug 12;53(6):419–27.

3. Friedman JH. Greedy function approximation: A gradient boosting machine. Ann Statist. 200110;29(5):1189–232.

4. Wolpert DH, Macready WG. Coevolutionary free lunches. IEEE Transactions on Evolutionary Computation. 2005 Dec;9(6):721–35. 





```python

```
