function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
%data = load('ex2data1.txt');
%[m,n] = size(X);
%Y = data(:,3);
%X = [ones(m,1) X];
%theta = zeros(n+1,1);
%a = X*theta;
% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
J =  (-y'*log(sigmoid(X*theta)) - (1-y)'*log(1-sigmoid(X*theta)))/m 
grad = zeros(size(theta));

grad =  ((sigmoid(X*theta) - y)'*X)'/m
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
