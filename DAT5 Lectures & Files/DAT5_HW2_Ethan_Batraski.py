# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from pandas import read_csv
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer

# <headingcell level=3>

# 1. Implement KNN classification, using the sklearn package

# <codecell>

df = pd.io.parsers.read_csv('data/iris_data.csv', header=None, names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name'], index_col=False)

# <codecell>

df.Name.unique()

# <codecell>

df = df.dropna()

# <codecell>

mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica' : 2}

# <codecell>

df_replaced = df.replace({'Name': mapping})

# <codecell>

df_replaced .columns

# <codecell>

X = df_replaced[[u'SepalLength', u'SepalWidth', u'PetalLength', u'PetalWidth']].values
y = df_replaced.Name.values

# <codecell>

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# <codecell>

myknn = KNeighborsClassifier(3).fit(X_train,y_train)

# <codecell>

myknn.score(X_test, y_test)

# <headingcell level=3>

# 2. Implement cross-validation for your KNN classifier.

# <codecell>

from sklearn.cross_validation import KFold

# generic cross validation function
def cross_validate(X, y, classifier, k_fold):
    
    #derive a set of (random) training and testing indices
    k_fold_indices = KFold(len(X), n_folds=k_fold,
                           indices=True, shuffle=True,
                           random_state=0)
    
    k_score_total = 0
    # for each training and testing slices run the classifier, and score the results
    for train_slice, test_slice in k_fold_indices :
        
        model = classifier(X[[ train_slice ]],
                           y[[ train_slice]])
        
        k_score = model.score(X[[ test_slice ]],
                              y[[ test_slice ]])
        
        k_score_total += k_score
                        
            
    # return the average accuracy
    return k_score_total/k_fold

# <codecell>

cross_validate(X, y, neighbors.KNeighborsClassifier(11).fit, 5)

# <headingcell level=3>

# 3. Use your KNN classifier and cross-validation code from (1) and (2) above to determine the optimal value of K (number of nearest neighbors to consult) for this Iris dataset. 

# <codecell>

int_max_k = 150
int_best_k = 1
float_max_score = 0.0
list_crossvalidation_scores = []
for int_index in range(150):
    int_k = int_index + 1
    float_score = cross_validate(X, y, neighbors.KNeighborsClassifier(int_k).fit, 19)
    if float_score > float_max_score:
        int_best_k = int_k
        float_max_score = float_score
    list_crossvalidation_scores.append(float_score)
    
               

# <codecell>

print int_best_k, float_max_score

# <headingcell level=3>

# 4. Using matplotlib, plot classifier accuracy versus the hyperparameter K for a range of K that you consider interesting. 

# <codecell>

plot(list_crossvalidation_scores)

# <headingcell level=4>

# It's interesting to see that there is a large drop off at ~65 and ~115 in accuracy. 

# <headingcell level=3>

# 5. OPTIONAL BONUS QUESTION: Using the value of K obtained in (3) above, vary the number of folds used for cross-validation across an interesting range, e.g. [ 2, 3, 5, 6, 10, 15]. How does classifier accuracy vary with the number of folds used? Do you think there exists an optimal number of folds to use for this particular problem? Why or why not?

# <rawcell>

# 2 folds - k(5) @ .96666
# 5 folds - k(11) @ .966666
# 10 folds - k(14) @ .966666
# 19 folds - k(19) @ .98026 (highest tested)
# 20 folds - k(19) @ .98025
# 21 folds - k(15) @ .97959
# 30 folds - k(19) @ .98
# 
# The classifer accuracy increases with the number folds until it reaches around 19, after which it seems to be the local maxima and accuracy diminishes. It seems there definitely exists an optimial number of folds to use for this particular problem. 
# 

# <codecell>


