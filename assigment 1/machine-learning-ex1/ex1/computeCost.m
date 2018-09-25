function J = computeCost(X, y, theta)
%v = load('ex1data1.txt');
%m = length(v);
%X = [ones(m,1) , v(:,1)];
%y = v(:,2);
m = length(y); 
%theta = zeros(2,1);
suma = sum((sum(theta' .* X , 2) - y).^2);
J = suma / (2*m)
end
