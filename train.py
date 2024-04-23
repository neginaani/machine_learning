from parameters import *
from cost import *
import numpy as np


class Train():
    def __init__(self , X , y, ITERATION , LEARNINGRATE , STOPPINGTHRESHOLD):
        self.X = X
        self.y= y
        self.ITERATION = ITERATION
        self.LEARNINGRATE= LEARNINGRATE
        self.STOPPINGTHRESHOLD= STOPPINGTHRESHOLD
       

    def train(self ):
        m, n =self.X.shape
        current_weight=np.random.rand(n,1)
        current_bias= np.random.rand(1,1)


        costs=list()
        weights=list()
        bias=list()
        previous_cost=None

        #estimation of optimal parameters
        for i in range(self.ITERATION):
            yhat= (self.X@current_weight)+ current_bias
            #cost function
            loss=Costfunction(yhat , self.y)
            current_cost=loss.Cost()

            #check to break
            if previous_cost==True and abs(current_cost-previous_cost)<=self.STOPPINGTHRESHOLD:
                break
            else:
                previous_cost=current_cost
                costs.append(current_cost)
                bias.append(current_bias)
                weights.append(current_weight)
                
                #calculaing the gradient
                weight_derivative= self.X.T@(yhat-self.y)
                bias_derivative= np.sum(yhat-self.y)

                #update parameters
                current_weight-= self.LEARNINGRATE *weight_derivative
                current_bias-= self.LEARNINGRATE*bias_derivative

        return weights ,bias ,  costs












            



            


    



