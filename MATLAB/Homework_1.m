%% Assignment #1
% Carter Kruse, July 12th

%% A) Compute the SVD of A.

% Matrix A
A = [0.16, 0.10; 
     0.17, 0.11;
     2.02, 1.29];

% Singular Value Decomposition
[U, S, V] = svd(A);

% Display Values
disp(U);
disp(S);
disp(V);

%% B) Compute the condition number of A.

% Using the built-in function that computes the condition number.
disp(cond(A));

% Using the computed SVD.
disp(S(1, 1) / S(2, 2));

% The values are the same.

%% C) Compute the data vector b, according to b = Ax + b_e.

% Vector X
x = [1; 2];

% Error
b_e = [0.01; -0.02; 0.03];

% Data Vector
b = A * x + b_e;

% Display Values
disp(b);

%% D) Compute the least squares solution x_ls.

% Least Squares Solution
x_ls = A \ b;

% Display Values
disp(x_ls);

% This is NOT close to the true solution x, which is a result of the noise
% introduced to the forward/inverse problem.

% It is reasonable to expect this kind of result because the matrix A is
% ill-conditioned, as indicated by the condition number.

% This means that the matrix A is nearly singular, and thus the effect of 
% noise/error is amplified when solving linear systems.

%% E) Use the truncated SVD.
% Aim: To arrive at a more reasonable solution to the inverse problem.
% The least squares solution x_ls can be expressed in terms of SVD.

% Maximum Singular Value
max_sigma = 1;

% Initialize (Solution) Vector
reconstruct = zeros(size(x));

% Apply Truncated SVD
for i = 1:max_sigma
    reconstruct = reconstruct + (U(:,i).' * b) / S(i, i) * V(:,i);
end

% Display Values
disp(reconstruct)
