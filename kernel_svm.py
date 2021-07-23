#!/usr/bin/env python
# coding: utf-8

# # Kernel SVM

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[2]:


dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[3]:


print(X)


# In[4]:


print(y)


# ## Splitting the dataset into the Training set and Test set

# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[6]:


print(X_train)


# In[7]:


print(X_test)


# In[8]:


print(y_train)


# In[9]:


print(y_test)


# ## Feature Scaling

# In[12]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Training the Kernel SVM model on the Training set

# In[13]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C= 0.5, gamma= 0.1, random_state = 0)
classifier.fit(X_train, y_train)


# # Applying Grid Search to find the best model and the best parameters

# In[15]:


from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)


# ## Predicting the Test set results 

# In[16]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Making the Confusion Matrix

# In[17]:


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
# accuracy_score(y_test, y_pred)


# ## Applying k-Fold Cross Validation# 

# In[18]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# ## Predicting a new result
# 

# In[19]:


print(classifier.predict(sc.transform([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])))


# In[ ]:




