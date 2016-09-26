
function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

function y = sig(x)
y = 1 / (1 + exp(-x));
end

if isa(z, 'scalar') == 1
    g = sig(z);
else
    g = arrayfun(@sig, z);
end

%for i = 1:size(z, 1)
%    for j = 1:size(z, 2)
%        fprintf('A single sigmoid: $%.2f\n', z(i:j));
%    end
%end
% =============================================================

end