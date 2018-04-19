import numpy as np 
from scipy.optimize import minimize 
from scipy.stats import norm 

class LinearRegression: # class
    
    def __init__(self, X, y): # constructor
        self.data = np.array(X) # label, target value, response variable
        self.y = np.array(y) # features, predictors
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
#np.random.seed(1) 
#X = np.random.uniform(0,10,100).reshape(50,2) 
#y = 3 + 1.3 * X[:,0] + 2.5 * X[:,1] + np.random.normal(0,3,50) 
# 
#lm1 = LinearRegression(X,y) lm1.summary() 
# 
#X_test = np.random.uniform(0,10,40).reshape(20,2) 
#y_test = 3 + 1.3 * X_test[:,0] + 2.5 * X_test[:,1] + np.random.normal(0,3,20) 
#print("Testing r^2:", lm1.score(X_test, y_test), "\n") 
# 
#X_new = np.array([[3,7], [6,1], [5,5]]) 
#print("Predictions:", lm1.predict(X_new)) 
# 
# 
#This should generate the following output: 
# 
#+-----------------------------+ 
#|  Linear Regression Summary  | 
#+-----------------------------+ 
#Number of training observations: 50 
#Coefficient Estimates:    [ 3.20059294  1.37664849  2.42832841] 
#Residual Standard Error: 2.40205928173 
#r-Squared: 0.917057007128 
#Log-Likelihood -113.763246836 
# 
#Testing r^2: 0.899105684284  
# 
#Predictions: [ 24.3288373   13.88881228  22.22547744] 
        
        
#np.random.seed(1) 
#X = np.random.uniform(0,10,150).reshape(50,3) 
#y = 7 - 2.1 * X[:,0] + 4.1 * X[:,1] + 3.5 * X[:,2] + np.random.normal(0,3,50) 
# 
#lm2 = LinearRegression(X,y) 
#lm2.summary() 
# 
#X_test = np.random.uniform(0,10,60).reshape(20,3) 
#y_test = 7 - 2.1 * X_test[:,0] + 4.1 * X_test[:,1] + 3.5 * X_test[:,2]  
#y_test += np.random.normal(0,3,20) 
#print("Testing r^2:", lm2.score(X_test, y_test), "\n") 
# 
#X_new = np.array([[3,7,1], [6,1,4], [5,5,5]]) 
#print("Predictions:", lm2.predict(X_new)) 
# 
# 
#This should generate the following output: 
# 
#+-----------------------------+ 
#|  Linear Regression Summary  | 
#+-----------------------------+ 
#Number of training observations: 50 
#Coefficient Estimates:    [ 6.50878947 -2.10607893  4.21017306  3.52534848] 
#Residual Standard Error: 2.93371474287 
#r-Squared: 0.969338511911 
#Log-Likelihood -123.760399177 
# 
#Testing r^2: 0.959919169606  
# 
#Predictions: [ 33.18711257  12.18388288  34.65600253] 