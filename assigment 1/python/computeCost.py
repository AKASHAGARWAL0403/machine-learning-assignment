import numpy as np
def computeCost(X,Y,theta):
    m = len(Y)
    cost = np.power((((X.dot(theta))) - Y),2)
    return np.sum(cost)/(2*m)

                
