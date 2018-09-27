import numpy as np
from scipy.io import loadmat

data = loadmat('ex8_movies.mat')

Y = data['Y']
R = data['R']

def cost(params,Y,R,num_features,learning_rate):
    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movie = Y.shape[0]
    num_user = Y.shape[1]
    
    X = np.matrix(np.reshape(params[:num_movie*num_features] , (num_movie,num_features))) #(1682,10)
    theta = np.matrix(np.reshape(params[num_movie*num_features:] , (num_user,num_features))) #(943,10)
    
    error = np.multiply(((X * theta.T) - Y) , R) #(1682,943)
    square_error = np.power(error,2)
    J = (1./2)*np.sum(square_error)
    
    J += ((learning_rate/2)*(np.sum(np.power(theta , 2))))
    J += ((learning_rate/2)*(np.sum(np.power(X,2))))
    
    X_grad = error*theta + (learning_rate*X)
    theta_grad = error.T * X + (learning_rate*theta)
    
    grad = np.concatenate((np.ravel(X_grad) , np.ravel(theta_grad)))
    return J,grad

movie_idx = {}  
f = open('movie_ids.txt')  
for line in f:  
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]
    movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])

ratings = np.zeros((1682, 1))

ratings[0] = 4  
ratings[6] = 3  
ratings[11] = 0  
ratings[53] = 4  
ratings[63] = 3  
ratings[65] = 3  
ratings[68] = 5  
ratings[97] = 2  
ratings[182] = 1  
ratings[225] = 5  
ratings[354] = 5

Y = np.append(Y , ratings , axis=1)
R = np.append(R , ratings != 0 , axis=1)

from scipy.optimize import minimize

movies = Y.shape[0]
users = Y.shape[1]
learning_rate = 10
num_features = 10

X = np.random.random(size=(movies,num_features))
theta = np.random.random(size=(users,num_features))
params = np.concatenate((np.ravel(X),np.ravel(theta)))

Ymean = np.zeros((movies,1))
Ynorm = np.zeros((movies,users))

for i in range(movies):
    idx = np.where(R[i,:] == 1)[0]
    Ymean[i] = Y[i,idx].mean()
    Ynorm[i,idx] = Y[i,idx] - Ymean[i]

fmin = minimize(fun=cost,x0=params,args=(Ynorm,R,num_features,learning_rate),
                 method='CG', jac=True, options={'maxiter': 100})

X = np.matrix(np.reshape(fmin.x[:movies*num_features] , (movies,num_features)))
theta = np.matrix(np.reshape(fmin.x[movies*num_features:] , (users,num_features)))

prediction = X * theta.T
my_pred = prediction[:,-1] + Ymean
sorted_preds = np.sort(my_pred, axis=0)[::-1]  
idx = np.argsort(my_pred, axis=0)[::-1]  
print("Top 10 movie predictions:")  
for i in range(10):  
    j = int(idx[i])
    print('Predicted rating of {0} for movie {1}.'.format(str(float(my_pred[j])), movie_idx[j]))
    print(j)


