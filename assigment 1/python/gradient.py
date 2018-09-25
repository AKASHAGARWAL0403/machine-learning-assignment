import numpy as np
import computeCost

def gradient(X,y,theta,alpha,iters):
    m = len(y)
    cost = np.zeros((iters,1))
    for i in range(iters):
        theta = theta - ((((X.dot(theta) - y).T).dot(X)).T)*(alpha/m)
        cost[i] = computeCost.computeCost(X,y,theta)
    return theta,cost