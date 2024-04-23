from train import *
import numpy as np
import pandas as pd 
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics


dataset=pd.read_csv("en.csv")
X= np.array(dataset.drop("Y1" , axis=1))
y=np.array(dataset[["Y1"]] )

X= (X - X.mean(axis=0)) / X.std(axis=0)


X_train, X_test, y_train , y_test=train_test_split(X, y , test_size=0.25)


def main():
    Xtr= X_train
    ytr=y_train
    Xts=X_test
    yts=y_test


    model= Train(X_train , y_train, ITERATION , LEARNINGRATE , STOPPINGTHRESHOLD)
    estimated_weights ,estimated_bias ,  estimated_costs=model.train()
    #print(f"the estimated_weights are {estimated_weights[-1]}\n and the estimated costs are {estimated_costs} " )
      #predictions
    y_hattr=(Xtr@estimated_weights[0])+ estimated_bias[0]
    y_hatts=(Xts@estimated_weights[0])+estimated_bias[0]

    
    print(f" the accuray of train is: {round(metrics.r2_score(y_hattr , ytr)*100,2)} and the accuracy of test is {round(metrics.r2_score(y_hatts , yts)*100,2)}")

    plt.plot([i for i in range(ITERATION)] , estimated_costs)
    plt.show()


  


    


if __name__=="__main__":
    main()


