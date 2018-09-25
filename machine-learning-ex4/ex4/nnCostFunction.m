function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Y_test = zeros(m,num_labels);
for i=1:m,
   Y_test(i,y(i)) = 1;
end

A1 = [ones(m, 1) X]; %size(X) = 5000 401
Z2 = A1*(Theta1');  % size(z1) = 5000 25
A2 = [ones(size(Z2,1) , 1) sigmoid(Z2)];  % size(a2) = 5000 26
Z3 = A2*(Theta2'); % size(z2) = 5000 10 
H = sigmoid(Z3);  % size(a3) = 5000 10
J = (1/m)*(sum(sum(((-Y_test) .* log(H)) - ((1-Y_test).*log(1-H))),2)) +  (lambda/(2*m))*(sum(sum((Theta1(:,[2:size(Theta1,2)])).^2,2)) + sum(sum((Theta2(:,[2:size(Theta2,2)])).^2)));

Sigma3 = H - Y_test;  %size(Sigma3) = 5000 10
Sigma2 = (Sigma3*Theta2 .* sigmoidGradient([ones(size(Z2,1),1) Z2]))(:,2:end);  %size(Sigma3) = 5000 25
Delta1 = Sigma2'*A1; % 25 401
Delta2 = Sigma3'*A2;  % 10 26
Theta1_grad = Delta1./m + (lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad = Delta2./m + (lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,2:end)];

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
