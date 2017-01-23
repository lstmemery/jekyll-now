
One of the more delightfully named theorems in data science is called "The No Free Lunch Theorem." It states "any two algorithms are equivalent when their performance is averaged across all possible problems."(4) If that's true, why did over half of the winning solutions for the data science competition website [Kaggle](https://www.kaggle.com) in 2015 contain XGBoost?(1) How does XGBoost even work? What does it look like in action?

I intend to answer these questions without asking you to do (too much) math. You will have to know about decision trees. A beautiful and intuitive introduction to decision trees can be found [here](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/).

## Gradient Boosted Trees Explained

Overfitting is the machine learning equivalent of cramming. If you only memorized the material without drawing out generalizations, you'll flub the final exam. Decision trees are great crammers, so data scientists have developed several techniques to make sure that they don't overfit.

One method is called boosting. Instead of training one strong learner (the overfitting decision tree) we instead train several weak (underfitting) learners in sequence. The trick is that the current tree is informed about which examples the previous tree got wrong. For boosting to work, the learners need to be incorrect in different ways. At the end of training, each tree votes on how each example should be classified. This way, the weaknesses found in some learners can be compensated by others. 

Adaptive boosting (AdaBoost) is a classic implementation of boosting. The first tree is punished equally for each misclassification. The next tree will be punished more harshly for getting the same examples wrong as the previous tree. This tactic will cause the new tree to prioritize making decisions that will correctly classify the previous hardest cases.

Gradient boosting is another flavor of boosting. The first tree fits the data in the typical fashion. The next tree tries to find the decisions that would minimize the error of the of the previous tree. It would be simple to minimize the error if we knew the mathematical function that described the data. If we knew the function, we wouldn't be trying to approximate it with decisions tree in the first place! We'll need a new tactic to minimize error without knowing the function.

Imagine you on top of a hill enveloped in fog. You can't see more than a step in front of you. What would be a reasonable way of finding the fastest way down? One way would be to put your foot out in each direction and feel the steepest way down. Now, step down the steepest slope you found. Repeat this process until you are at the bottom. Data scientists call this algorithm gradient descent. In this context, we call it the Gradient Boosted Trees algorithm.

<a href="http://vis.supstat.com/2013/03/gradient-descent-algorithm-with-r/">
<figure>
    <img src='{{ site.baseurl }}/assets/gradient_descent.gif' alt='Gradient Descent' />
    <figcaption>Yu L. Demonstration of the Gradient Descent Algorithm. Vistat [Internet]. 2013 Mar 24 [cited 2017 Jan 21]; Available from: http://vis.supstat.com/2013/03/gradient-descent-algorithm-with-r/</figcaption>
</figure>
</a>
## What makes XGBoost EXTREME?

The first proposal for gradient boosted trees was published in 2001.(3) What has XGBoost boost done to improve the algorithm?

One common misconception I've seen is that XGBoost is somehow more accurate than other implementations of the gradient boosted tree algorithm. It's not true! The authors of the algorithm found that XGBoost had roughly the same error rate as the Sci-Kit Learn implementation.

### XGBoost is not extremely accurate; XGBoost is extremely fast

1. **XGBoost has sparsity-awareness:** Boosted trees work especially well on categorical feature (e.g. Was this person born England?). In many real-world data sets, a categorical feature column will be mostly zeros. When deciding where to split, XGBoost has indices of the non-zero data are and only needs to look at those entries.
2. **XGBoost is parallelizable:** The most time-consuming step for boosted tree algorithms is sorting continuous features (e.g. How far do you drive to work each day?). The sparse data structure and clever implementation allow XGBoost sort columns independently. This way, the sorting work can be divided up between parallel threads of the CPU. 
3. **XGBoost can split approximately:** To find the most efficient cut on a continuous feature, gradient boosted trees need to keep all of the data in memory at the same time to sort it. This is not a problem for small datasets, but it becomes impossible when you have more data than RAM. XGBoost has the ability to bin these numbers in rough order instead of sorting them entirely. The authors of the XGBoost paper show that, with enough bins, you get approximately the same performance as with the exact split in a fraction of the time.

## An illustration

Let's take a look at what XGBoost can do. I'll be working with the training data from the Ames Housing Dataset, which can be found [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

To prepare this data, I used Human Analog's code graciously provided via a [Kaggle kernel](https://www.kaggle.com/humananalog/house-prices-advanced-regression-techniques/xgboost-lasso/code).  

This script cleans up the dataset, get's it into an XGBoost-readable format and engineers new features from existing data. For example, he adds a feature that describes whether the house was last renovated the year it was sold.


```python
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, \
    AdaBoostRegressor, RandomForestRegressor



X = pd.read_csv('../../results/munged_training.csv')
y = pd.read_csv('../../results/munged_labels.csv')

# Make a validation set
X_train, X_validation, y_train, y_validation = train_test_split(X, 
                                                                y, 
                                                                random_state=1848)
```

```python
# Sci-Kit Learn's Out of the Box Gradient Tree Implementation
sklearn_boost = GradientBoostingRegressor(random_state=1849)
sklearn_boost.fit(X_train, y_train.values.ravel())
print('Training Error: {:.3f}'.format(1 - sklearn_boost.score(X_train, 
                                                              y_train)))
print('Validation Error: {:.3f}'.format(1 - sklearn_boost.score(X_validation, 
                                                                y_validation)))
%timeit sklearn_boost.fit(X_train, y_train.values.ravel())
```

    Training Error: 0.031
    Validation Error: 0.110
    1 loop, best of 3: 1.23 s per loop



```python
# XGBoost
xgb_boost = xgb.XGBRegressor(seed=1850)
xgb_boost.fit(X_train, y_train.values.ravel())
print('Training Error: {:.3f}'.format(1 - xgb_boost.score(X_train, 
                                                          y_train)))
print('Validation Error: {:.3f}'.format(1 - xgb_boost.score(X_validation, 
                                                            y_validation)))
%timeit xgb_boost.fit(X_train, y_train.values.ravel())
```

    Training Error: 0.038
    Validation Error: 0.111
    1 loop, best of 3: 443 ms per loop



```python
# Sci-Kit Learn's Adaptive Boosting
ada_boost = AdaBoostRegressor(random_state=1851)
ada_boost.fit(X_train, y_train.values.ravel())
print('Training Error: {:.3f}'.format(1 - ada_boost.score(X_train, 
                                                          y_train)))
print('Validation Error: {:.3f}'.format(1 - ada_boost.score(X_validation, 
                                                            y_validation)))
%timeit ada_boost.fit(X_train, y_train.values.ravel())
```

    Training Error: 0.126
    Validation Error: 0.196
    1 loop, best of 3: 729 ms per loop



```python
# Random Forest: A fast, non-boosting algorithm
random_forest = RandomForestRegressor(random_state=1852)
random_forest.fit(X_train, y_train.values.ravel())
print('Training Error: {:.3f}'.format(1 - random_forest.score(X_train, 
                                                              y_train)))
print('Validation Error: {:.3f}'.format(1 - random_forest.score(X_validation, 
                                                                y_validation)))
%timeit random_forest.fit(X_train, y_train.values.ravel())
```

    Training Error: 0.024
    Validation Error: 0.128
    1 loop, best of 3: 398 ms per loop


First, let's take a look at look at the validation error. The Sci-Kit Learn implementation and XGBoost are nearly identical and both beat AdaBoost and RandomForest with their default hyperparameters. However, XGBoost really shines in the speed of execution. XGBoost is three times as fast as the Sci-Kit Learn implementation and nearly as quick as RandomForest.

The fact that XGBoost is generally accurate and fast makes it an excellent tool for evaluating feature engineering. If you think of a new feature, add it in, spin up XGBoost, cross-validate and see if the new feature improved the accuracy of the model. Because boosted trees are robust to overfitting, you can continue to add features until you run out of ideas. 

XGBoost isn't used everywhere because it's the best at everything (although it performs very well in many cases). It's used everywhere because it's a sane default for when you are poking around the data.

## Bibliography

1. Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. arXiv:160302754 [cs]. 2016;785–94. 

2. Mayr A, Binder H, Gefeller O, Schmid M. The Evolution of Boosting Algorithms - From Machine Learning to Statistical Modelling. Methods of Information in Medicine. 2014 Aug 12;53(6):419–27.

3. Friedman JH. Greedy function approximation: A gradient boosting machine. Ann Statist. 200110;29(5):1189–232.

4. Wolpert DH, Macready WG. Coevolutionary free lunches. IEEE Transactions on Evolutionary Computation. 2005 Dec;9(6):721–35. 




Do you have a data science topic you'd like me to cover? Shoot me an email at m{dot}emery{at}alumni{dot}ubc{dot}ca.
