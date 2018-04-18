import numpy as np 
from scipy.optimize import minimize 
from scipy.stats import norm 

class LinearRegression:#class
    
    def __init__(self, X, y):#constructor, method cuz it has self as an argument
        self.data = np.array(X)
        self.y = np.array(y) #Contains the training labels
        self.n_observations  = len(y) # The number of training observations. 
        
        def find_sse(beta):  #doesnt need self cuz its not method proposed model
            y_hat = beta[0] + np.sum(beta[1:] * X, axis=1) #or this way
#            y_hat = []
#            for i in range(X.shape[0]):
#                temp = beta[0] + np.sum(beta[1:]*X[i,:])
#                y_hat.append(temp)
#            y_hat = np.array(y_hat)
            residuals = y - y_hat
            self.sse = np.sum(residuals**2)
            return self.sse
        beta_guess = np.zeros(X.shape[1]+1)
        min_results = minimize(find_sse, beta_guess)
        self.coefficients = min_results.x # Contains the parameter estimates: 𝛽0 ̂, 𝛽1 ̂,… , 𝛽𝑝 ̂
        
#        def predict(X): #this is predict inner function and call this just predict(X) not self.predict
#            # this method should return a numpy array of predicted y values. 
#            # Convert X to a NumPy array 
#            X = np.array(X)
#            yHatForNewData = []
#            for j in range(X.shape[0]):
#                tempo = self.coefficients[0] + np.sum(self.coefficients[1:]*X[j,:])
#                yHatForNewData.append(tempo)
#            yHatForNewData = np.array(yHatForNewData)
#            return yHatForNewData 
#        
        self.y_predicted = self.predict(X)
        self.residuals = self.y - self.y_predicted 
        self.sse = np.sum((self.residuals)**2) 
        self.sst = np.sum((self.y-(np.mean(self.y)))**2)
        self.r_squared = 1 - self.sse / self.sst
        self.rse = (self.sse/(self.n_observations-2))**0.5 
#        sstd = (self.sse / (self.n_observations-1)) ** 0.5 
#        sumForLoglik = 0
        self.loglik = np.sum(np.log(norm.pdf(self.residuals,0,self.rse))) #or this way
#        for k in range(len(self.residuals)):
#            sumForLoglik += np.log(norm.pdf(self.residuals[k],0,sstd))
#        self.loglik = sumForLoglik
#        print(math.log(2.7)) # prints 0.9932517730102834 = math.log(2.7, e)
        
    def predict(self, X): 
        # this method should return a numpy array of predicted y values. 
        # Convert X to a NumPy array 
        X = np.array(X)
        yHatForNewData = []
        for j in range(X.shape[0]):
            tempo = self.coefficients[0] + np.sum(self.coefficients[1:]*X[j,:])
            yHatForNewData.append(tempo)
        yHatForNewData = np.array(yHatForNewData)
        return yHatForNewData
        
    def score(self, X, y):
        X = np.array(X)
        y = np.array(y) 
        e = y-self.predict(X) #residuals are equal to y minus y hat, y hats are predicted y
        sseForNewData = np.sum(e**2)
        sstForNewData = np.sum((y-np.mean(y))**2)        
        rSquaredForNewData = (1 - (sseForNewData / sstForNewData))
        return rSquaredForNewData #these shouldnt be attributes cuz we dont wanna store them.
#       also we wanna tell sseForNewData and self.sse, which is for sse for training data.
    
    def summary(self):
        print('+-----------------------------+')
        print('|  Linear Regression Summary  |')
        print('+-----------------------------+')
        print('Number of training observations:',self.n_observations)
        print('Coefficient Estimates:',self.coefficients)
        print('Residual Standard Error:',self.rse)
        print('r-Squared:',self.r_squared)
        print('Log-Likelihood',self.loglik)#differ from what it should be
#//////////////////  
np.random.seed(1)
X = np.random.uniform(0,10,150).reshape(50,3)
y = 7 - 2.1 * X[:,0] + 4.1 * X[:,1] + 3.5 * X[:,2] + np.random.normal(0,3,50)
lm2 = LinearRegression(X,y)
lm2.summary()
X_test = np.random.uniform(0,10,60).reshape(20,3)
y_test = 7 - 2.1 * X_test[:,0] + 4.1 * X_test[:,1] + 3.5 * X_test[:,2]
y_test += np.random.normal(0,3,20)
print("Testing r^2:", lm2.score(X_test, y_test), "\n")
X_new = np.array([[3,7,1], [6,1,4], [5,5,5]])
print("Predictions:", lm2.predict(X_new))

#X = np.array([[12,4],[14,3],[16,6],[20,5],[24,2]])
#y = np.array([50, 53, 67, 70, 63])


