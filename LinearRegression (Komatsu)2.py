import numpy as np 
from scipy.optimize import minimize 
from scipy.stats import norm 
import math   # look like this?

class LinearRegression:#class
    
    def __init__(self, X, y):#constructor, method cuz it has self as an argument
        self.data = np.array(X)
        self.y = np.array(y) #Contains the training labels
        self.n_observations  = len(y) # The number of training observations. 
        
        def find_sse(beta): #doesnt need self cuz its not method proposed model
#            individualResidual = []
            y_hat = []
            betaNp = np.array(beta)
#            sum = 0
            global predictedValue
            predictedValue = []
            for i in range(X.shape[0]):
                temp = beta[0] + np.sum(beta[1:]*X[i,:])
                y_hat.append(temp)
                predictedValue.append(betaNp[0] + betaNp[1] * self.data[i,0] + betaNp[2] * self.data[i,1])
#                individualResidual.append(self.y[i] - predictedValue[i])
#                sum += (individualResidual[i])**2
#                self.sse = sum
            y_hat = np.array(y_hat)
            residuals = y - y_hat
            self.sse = np.sum(residuals**2)
            return self.sse
#        # Now we calculate y_hat and sse.
#y_hat = beta[0] + np.sum(beta[1:] * X, axis=1)
#residuals = y - y_hat
#sse = np.sum(residuals**2)
#
## Print the results
#print('y_hat = ', y_hat)
#print('residuals = ', residuals)
#print('sse = ', sse)
        beta_guess = np.zeros(X.shape[1]+1)
        min_results = minimize(find_sse, beta_guess)
        self.coefficients = min_results.x # Contains the parameter estimates: 𝛽0 ̂, 𝛽1 ̂,… , 𝛽𝑝 ̂
        
        def predict(self, X): #is this ok?
            # this method should return a numpy array of predicted y values. 
            # Convert X to a NumPy array 
            X = np.array(X)
            return np.array(predictedValue)
        
        self.y_predicted = self.predict(X) #or np.array(predictedValue)?
        self.residuals = self.y - self.y_predicted 
        self.sse = np.sum((self.residuals)**2) #or using find_sse?? but what is beta in this case?
        self.sst = np.sum((self.y-(np.mean(self.y)))**2)
        self.r_squared = 1 - self.sse / self.sst
        self.rse = (self.sse/(self.n_observations-2))**0.5
        sstd = (self.sse / (self.n_observations-1)) ** 0.5
#        for j in range(self.n_observations): #this spits an error
#            self.loglik = np.sum(math.log(np.random.normal(self.residuals[j],0,sstd[j])))#plug in self.residuals??
#        print(math.log(2.7)) # prints 0.9932517730102834 = math.log(2.7, e)
        
    def predict(self, X):  #is this ok?
        # this method should return a numpy array of predicted y values. 
        # Convert X to a NumPy array 
        X = np.array(X)
        return np.array(predictedValue)
    
    def score(self, X, y):
        X = np.array(X)
        y = np.array(y) # you do like this on SplitClassifier?
        self.sse = np.sum((self.residuals)**2) #or using find_sse?? but what is beta in this case?
        self.sst = np.sum((self.y-(np.mean(self.y)))**2)        
        self.rSquared = 1 - (self.sse / self.sst)
        
    def summary(self):
        print('+-----------------------------+')
        print('|  Linear Regression Summary  |')
        print('+-----------------------------+')
        print('Number of training observations:',self.n_observations)
        print('Coefficient Estimates:',self.coefficients)
        print('Residual Standard Error:')  # havent learned?
        print('r-Squared:',self.r_squared)
#        print('Log-Likelihood',self.loglik)#this spits an error
#//////////////////  
np.random.seed(1) 
X = np.random.uniform(0,10,100).reshape(50,2) 
#print(X)
y = 3 + 1.3 * X[:,0] + 2.5 * X[:,1] + np.random.normal(0,3,50) # 3 + beta1 times x +beta2 times x??
#print(y)# the training labels.  why doesnt it like [0,1]?
#        
        
#X = np.array([[12,4],[14,3],[16,6],[20,5],[24,2]])
#y = np.array([50, 53, 67, 70, 63])
lm1 = LinearRegression(X,y)
lm1.summary()
print("x", X)
print()
print("y", y)

