# Machine-Learning-Training

Author: Mohammadreza Ebrahimi  
Email: [m.reza.ebrahimi1995@gmial.com](mailto:m.reza.ebrahimi1995@gmial.com)

I have created a portfolio for training machine learning. 

By study this repository you will be laern how to develope a two main model training in machine learning which are   
- Regression
- Classification

Where those are subset of supervised learning.  

- Project 1: [Predict House Price in California](https://github.com/mohammadreza-ebrahimi/Machine-Learning-Training/tree/main/House-Regression)
- Project 2: [Classify Digit Number](https://github.com/mohammadreza-ebrahimi/Machine-Learning-Training/tree/main/Digit-Classifier)  

#### Project 1  


In this project I used data from USA statistics dataset in order to analysis house data in california and build a model to predict the house value. Following steps 
is needed  

- Getting some primary information about data
- Visualiaing data and see the correlation between features
- Splitting train set and test set by stratifying them
- Chooing the best model which is fitted to training data
- Evaluate cost function for each selected model

#### Project 2

In project 2 I used data from MNIST dataset, a set of data included handwriting digit numbers. The aim pf this project is to predict which sets of array is related to which class. Following steps explained  

- Loading data from MNIST
- Getting some information about data
- Visualizing some of them as instance
- Splitting train set and test set
- Fixing target, divide traget column to two classes of `True` and `False` or `Yes` and `No`.
- Training data with **SGD Classifier**
- Evaluating **Percision** and **Recall** for this model.

After above steps, I proceed in multi class model training with following steps  
- This time, It is not required to fix the tearget due to multi class model training
- Choosing appropriate model, again SGD classifier, by default it is **OvA**. I also examined **OvO** method.
- Test some other models
- Evaluating **Percision** and **Recall**
- Visualizing **Confusion Matrix** in order to analysis errors


If you have any question or suggestion please [contact](mailto:m.reza.ebrahimi1995@gmial.com) me. 
I will be happy.
