


import numpy as np
import pandas as pd
import ipywidgets as widgets
from ipywidgets import Layout
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def logistic_regression(beta0, beta1, sl, ss, p, sd):

    x = np.array([2, 5, 6, 7, 8, 10])
    y = np.array([0, 0, 1, 0, 1, 1])

    lm = LinearRegression()
    lm.fit(x.reshape(6,1),y)
    xticks = np.linspace(0,10,100)
    y_line= lm.predict(xticks.reshape(100,1))

    def sigmoid(z):
        return 1 / (1 + np.exp(-(beta0 + beta1*z)))
    
    #z = beta0 + beta1*xticks
    #sig = 1 / (1 + np.exp(-z))

    plt.close()
    plt.rcParams["figure.figsize"] = [8,6]
    plt.scatter(x[y==1],y[y==1], c='b')
    plt.scatter(x[y==0],y[y==0], c='r')
    if sl: plt.plot(xticks, y_line)
    if ss: plt.plot(xticks, sigmoid(xticks))
    if p:
        plt.plot([2,2],[sigmoid(2),1],c='r')
        plt.plot([5,5],[sigmoid(5),1],c='r')
        plt.plot([7,7],[sigmoid(7),1],c='r')
        plt.plot([6,6],[0,sigmoid(6)],c='b')
        plt.plot([8,8],[0,sigmoid(8)],c='b')
        plt.plot([10,10],[0,sigmoid(10)],c='b')
    if sd: plt.plot([-beta0/beta1,-beta0/beta1],[0,1],linestyle=':', c='orange')    
    plt.show()
    
    prob = (1-sigmoid(2))*(1-sigmoid(5))*(1-sigmoid(7))*sigmoid(6)*sigmoid(8)*sigmoid(10)
    
    if p:
        print('P[Y=0|X=2] * P[Y=0|X=4] * P[Y=1|X=6] * P[Y=0|X=7] * P[Y=1|X=8] * P[Y=1|X=10]')
        print('  =', round(1-sigmoid(2),4),'*',round(1-sigmoid(4),4),'*',
              round(sigmoid(6),4),'*',round(1-sigmoid(7),4),'*',
              round(sigmoid(8),4),'*',round(sigmoid(10),4))
        print('  =', round(prob,5))
    

_ = widgets.interact(logistic_regression,
                     beta0=widgets.FloatSlider(min=-10,max=0,step=0.05,value=-4,
                                               continuous_update=False),
                     beta1=widgets.FloatSlider(min=0,max=2,step=0.01,value=0.75,
                                               continuous_update=False),
                     sl=widgets.Checkbox(value=False,description='Show Line',
                                         disable=False),
                     ss=widgets.Checkbox(value=False,description='Show Sigmoid',
                                         disable=False),
                     p=widgets.Checkbox(value=False, description='Show Probabilities',
                                        disable=False),
                     sd=widgets.Checkbox(value=False, description='Show Decision Boundary',
                                        disable=False))
