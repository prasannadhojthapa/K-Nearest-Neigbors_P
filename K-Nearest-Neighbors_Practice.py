#!/usr/bin/env python
# coding: utf-8

# K-Nearest Neighbors
# 

# In[1]:


import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

#Install Kaggle
#pip install kaggle


# In[3]:


import mglearn

mglearn.plots.plot_knn_classification(n_neighbors=1)


# In[4]:


mglearn.plots.plot_knn_classification(n_neighbors=3)


# Now let’s look at how we can apply the k-nearest neighbors algorithm using scikitlearn.

# In[5]:


from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# Next, we import and instantiate the class. This is when we can set parameters, like the
# number of neighbors to use.

# In[7]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)


# In[9]:


#We fit the classifier using the training set. for `KNEighborsClassifier` 
#This means storing the dataset, so we can compute neighbors during prediction:

clf.fit(X_train, y_train)

print("Test set predictions: {}".format(clf.predict(X_test)))


# To evaluate how well our model generalize. Use `score` method. 

# In[10]:


print("Test set accuracyL {:2f}".format(clf.score(X_test, y_test)))


# In[ ]:




