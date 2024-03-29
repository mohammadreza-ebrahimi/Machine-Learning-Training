# Machine-Learning-Training

Author: Mohammadreza Ebrahimi  
Email: [m.reza.ebrahimi1995@gmial.com](mailto:m.reza.ebrahimi1995@gmial.com)  
***

I have created a portfolio for training machine learning. 

By study this repository you will laern how to develope two main models training in machine learning which are   
- Regression  
- Classification

Where those are subset of supervised learning.  
***

- Project 1: [Predict House Prices in California](https://github.com/mohammadreza-ebrahimi/Machine-Learning-Training/tree/main/House-Regression)
- Project 2: [Predict Gross Domestic Product (GDP) of a country](https://github.com/mohammadreza-ebrahimi/Machine-Learning-Training/tree/main/world-happiness)
- Project 3: [Classify Digit Numbers](https://github.com/mohammadreza-ebrahimi/Machine-Learning-Training/tree/main/Digit-Classifier)  
- Project 4: [Classify Mobile Prices](https://github.com/mohammadreza-ebrahimi/Machine-Learning-Training/tree/main/mobile)
- Project 5: [Classify Sky Objects](https://github.com/mohammadreza-ebrahimi/Machine-Learning-Training/tree/main/sky)

#### Project 1  
#### Project 2
![hr-versus-runs-regression](hr-versus-runs-regression.png)

In project 1, I analyzed house data in California and developed a model to forecast house value using data from the USA statistics dataset. The steps listed below are required. In addition, in project 2 (supervised, regression, similar to project 1), I used GDP data for worldwilde to forecast each country's GDP per capita.

- Obtaining some basic data information
- Visualizing data and checking the relations among characteristics 
- Splitting the train and test set by stratifying them
- Choosing the best model that fits the training data 
- Evaluating the cost function for each model that is chosen

#### Project 3  
#### Project 4
#### Project 5
![classification-algorithm-in-machine-learning](classification-algorithm-in-machine-learning.png)  

In project 3 I used data from the MNIST dataset, a set of data that includs handwriting digit numbers. This project aims to to predict which set groupsare related to which class. 
In project 4, I analyze data about mobiles and their **price range**. Project 5 is about sky objects classification. Briefly, all of them are used to **classify binary or multi-labels**, the following steps are explained  

- Loading data from MNIST
- Getting some information about data
- Visualizing some of them as instance
- Splitting train and test set
- Fixing target, divide traget column by two classes of `True` and `False` or `Yes` and `No` or even `1` and `0`. 
- Training data with **SGD Classifier**
- Evaluating **Percision** and **Recall** and **F1** for this model.

After the steps above, I proceed in multi-class model training by the following steps  
- This time, It is not required to fix the target due to multi-class model training
- Choosing the appropriate model, again SGD classifier, by default it is **OvA**. I also examined the **OvO** method.
- Test some other models
- Evaluating **Precision**, **Recall** and **F1**. 
- Visualizing **Confusion Matrix** to analyze errors

Some important concepts and definitions are from this [book](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646) written by ***Aurélien Géron***  
If you have any question or suggestion please [contact](mailto:m.reza.ebrahimi1995@gmial.com) me. 
I will be happy.
