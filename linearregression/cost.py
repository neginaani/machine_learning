import numpy as np 

class Costfunction:
    def __init__(self , yhat, yreal):
        self.yhat = yhat
        self.yreal=yreal
        
    def Cost(self):
        m,_=self.yreal.shape
        mse= (1/(2*m))* np.sum((self.yhat - self.yreal) **2)
        return mse




