import computeCost
import numpy as np

def gradient(X,Y,theta,alpha,iters):
    m = len(Y)
    cost = np.zeros((iters,1))
    for i in range(iters):
        theta = theta - ((((X.dot(theta) - Y).T).dot(X)).T)*(alpha/m)
        cost[i] = computeCost.computeCost(X,Y,theta)
    return theta,cost
