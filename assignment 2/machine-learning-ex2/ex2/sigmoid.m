function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
%data = load('ex2data1.txt');
%X = data(:,[1,2]);
%[m,n] = size(X);
%Y = data(:,3);
%X = [ones(m,1) X]
%theta = zeros(n+1 , 1);
%a = X*theta;
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).





% =============================================================
z = z * -1;
g  = 1 ./ ( exp(z) + 1);
end;