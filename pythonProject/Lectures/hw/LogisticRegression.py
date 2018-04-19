# Takaaki Komatsu
import numpy as np
from scipy.optimize import minimize 
import pandas as pd

class LogisticRegression:
    def __init__(self, X, y):#constructor
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_observations  = len(y) # The number of training observations. 
        self.classes = np.unique(self.y)#sorted(set(self.y)) # like [0,1] use np.unique()     
        
        def find_neg_loglik(beta):
            z = beta[0] + np.sum(beta[1:] * self.X, axis=1)
            p = 1 / (1 + np.exp(-z))            
            pi = np.where(self.y == self.classes[1], p, 1 - p)
            pi = np.where(pi == 0, 1e-100, pi)# without this one, have an error cuz we wanna avoid zero
            loglik = np.sum(np.log(pi))
            negativeLoglik = (-1) * loglik
            return negativeLoglik
        
        np.seterr(all='ignore')        
        beta_guess = np.zeros(X.shape[1]+1)      
        min_results = minimize(find_neg_loglik, beta_guess)
        np.seterr(all='warn')
        self.coefficients = min_results.x 
               
        self.y_predicted = self.predict(self.X) #correct?
        self.accuracy = np.sum(self.y_predicted==self.y) / self.n_observations
        self.loglik = (-1) * find_neg_loglik(self.coefficients)

    def predict_proba(self, X):
        self.npX = np.array(X)
        z2 = self.coefficients[0] + np.sum(self.coefficients[1:] * self.npX, axis=1)
        p2 = 1 / (1 + np.exp(-z2))     
        return p2
    
    def predict(self, X, t=.5):
        X = np.array(X)
        class1 = np.where(self.predict_proba(X) >= t, self.classes[1], self.classes[0])
        return class1
    
    def summary(self):
        print('+-------------------------------+')
        print('|  Logistic Classifier Summary  |')
        print('+-------------------------------+')
        print('Number of training observations:', self.n_observations)
        print('Coefficient Estimates:', self.coefficients)
        print('Log-Likelihood:', self.loglik)
        print('Accuracy:', self.accuracy, '\n') 

    def score(self, X, y, t=.5):
        X = np.array(X)
        y = np.array(y)
        accuracy = np.sum(self.predict(X, t) == y) / len(y)
        return accuracy
    
    def confusion_matrix(self, X, y, t=.5):
        X = np.array(X)
        y = np.array(y)  
        predictedLabels = self.predict(X, t)
        tp = np.sum((predictedLabels == self.classes[1]) & (y == self.classes[1]))
        fp = np.sum((predictedLabels == self.classes[1]) & (y == self.classes[0]))
        tn = np.sum((predictedLabels == self.classes[0]) & (y == self.classes[0]))
        fn = np.sum((predictedLabels == self.classes[0]) & (y == self.classes[1]))
        cm = pd.DataFrame([[tn,fp],[fn,tp]])
        cm.columns = ["Pred_0", "Pred_1"]
        cm.index = ["True_0", "True_1"]
        print('Class 0:', self.classes[0])
        print('Class 1:', self.classes[1], '\n')   
        print(cm, '\n')

from sklearn.datasets import make_classification 
 
np.random.seed(2) 
X, y = make_classification(n_samples = 237, n_features = 5, n_informative =3, n_redundant = 0, n_classes = 2) 
 
logreg_mod = LogisticRegression(X,y) 
logreg_mod.summary() 
 
logreg_mod.confusion_matrix(X,y) 
 
logreg_mod.confusion_matrix(X,y, t = 0.3) 