import numpy as np 
from scipy.optimize import minimize 
from scipy.stats import norm 

class LinearRegression: # class
    
    def __init__(self, X, y): # constructor
        self.data = np.array(X) # label, target value, or response variable
        self.y = np.array(y) # features or predictors
        self.n_observations  = len(y) # The number of training observations. 
        
        def find_sse(beta): # inner function that finds sse of the training data
            y_hat = beta[0] + np.sum(beta[1:] * self.data, axis=1) # y_hat is predicted y value = beta0+beta1*x(1)+...
            residuals = self.y - y_hat # residuals are value of y - predicted y value
            self.sse = np.sum(residuals**2)
            return self.sse # sse = sum of the square of the resideuals
        
        beta_guess = np.zeros(self.data.shape[1]+1)
        min_results = minimize(find_sse, beta_guess)
        self.coefficients = min_results.x 
        self.y_predicted = self.predict(self.data) 
        self.residuals = self.y - self.y_predicted 
        self.sse = np.sum((self.residuals)**2) # sse = sum of the square of the resideuals
        self.sst = np.sum((self.y - (np.mean(self.y)))**2) 
        self.r_squared = 1 - self.sse / self.sst #minimizing SSE is equivalent to maximizing r^2
        self.rse = (self.sse/(self.n_observations - 2))**0.5 # √𝑆𝑆𝐸/(𝑛 −2) # residual standard error
        self.loglik = np.sum(np.log(norm.pdf(self.residuals,0,self.rse))) 
        #loglik = ln[𝑓(e hat 1)] +ln[f(e hat 2)] + ...
        
    def predict(self, X): 
        X = np.array(X) # Convert X to a NumPy array 
        yHatForNewData = self.coefficients[0] + np.sum(self.coefficients[1:] * X, axis=1) 
        return yHatForNewData # predict y values based on the optimal coefficients
        
    def score(self, X, y):
        X = np.array(X)
        y = np.array(y) 
        e = y-self.predict(X) 
        sseForNewData = np.sum(e**2)
        sstForNewData = np.sum((y-np.mean(y))**2)        
        rSquaredForNewData = (1 - (sseForNewData / sstForNewData))
        return rSquaredForNewData 
    
    def summary(self):
        print('+-----------------------------+')
        print('|  Linear Regression Summary  |')
        print('+-----------------------------+')
        print('Number of training observations:',self.n_observations)
        print('Coefficient Estimates:',self.coefficients)
        print('Residual Standard Error:',self.rse)
        print('r-Squared:',self.r_squared)
        print('Log-Likelihood',self.loglik)