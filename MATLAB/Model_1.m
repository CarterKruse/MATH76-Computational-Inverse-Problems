%% Binary Signal Recovery
% Computational Inverse Problems, Math 076 (Summer 2023)
% Carter Kruse

clearvars;

% Reproducibility
rng('Default');

% Each computational section will build on the previous. When MATLAB
% specific code is given, tools such as ChatGPT or other
% language-conversion programs may be used to translate the code into a
% different coding language.

%% Model 1) Random Matrix

% Let N = 100.
N = 100;

%% Set-Up
% Generate a matrix 'A' in R^{N x N} such that 'A' is a matrix of normally 
% distributed random numbers with mean '0' and standard deviation '1'.

% In MATLAB, this can be accomplished using the 'randn' function.
A = randn(N, N);

% Normalize A
A = A ./ norm(A, 2);

%% SVD
% Compute the SVD of 'A'.

% Singular Value Decomposition
[U, S, V] = svd(A);

% Display Values
% disp(U);
% disp(S);
% disp(V);

%% Condition Number
% Compute the condition number of 'A'.

% Using the built-in function that computes the condition number.
disp(cond(A));

% Visualizing the matrix.
figure;
imagesc(A);
colorbar;

% Title
title('Matrix A');

%% Data Signal
% Generate a vector 'x' using the following code.

x = zeros(N, 1);
x(10:20) = 1;
x(30:60) = 1;
x(70:85) = 1;
x(90:95) = 1;

% This should result in an 'x' that is piecewise-constant.

% Plot
figure;
plot(x);

% Title
title('Vector X');

% Limits
ylim([-0.5 1.5]);

%% Computation
% Compute the data vector 'b', according to 'b = Ax + ε', where 'ε' is Gaussian 
% white noise with standard deviation 'η = 1'.

% Additive Gaussian Noise
eta = 0.2; epsilon = eta * randn(N, 1);

% Compute Forward Problem
b_exact = A * x;

% Add Noise
b = b_exact + epsilon;

% Display Values
% disp(b);

% Plot
figure;
plot(b_exact);

% Title
title('Data');

hold on;
    plot(b, 'r');
hold off;

legend('Exact', 'Observed')

%% Plots

figure;

subplot(1, 2, 1);
plot(x);

% Title
title('X');

% Limits
ylim([-0.5 1.5]);

subplot(1, 2, 2);
plot(b);

% Title
title('Observed');

%% Least Squares
% Compute the least squares solution 'x_ls'.

x_ls = A \ b;
% x_ls = inv(A' * A) * A' * b;

% Display Values
disp(x_ls);

% Plot
figure;
plot(x_ls);

% Title
title('Least Squares');

% This is NOT close to the true solution 'x', which is a result of the noise
% introduced to the forward/inverse problem.

% It is reasonable to expect this kind of result because the matrix 'A' is
% ill-conditioned, as indicated by the condition number.

% This means that the matrix 'A' is nearly singular, and thus the effect of 
% noise/error is amplified when solving linear systems.

%% Plot SVD

figure;
plot(log(diag(S)), '*');

% Title
title('Picard Log Plot');

% Multiple Plots
hold on;
    plot(log(abs(U' * b)), 'o');
    plot(log(abs(U' * b) ./ diag(S)), 'x');
hold off;

% Legend
legend('\sigma_i', '|u_i^Tb|','|u_i^Tb| / \sigma_i', 'location', 'northwest');

%% Truncated SVD
% Aim: To arrive at a more reasonable solution to the inverse problem.
% The least squares solution 'x_ls' can be expressed in terms of SVD.

% Maximum Singular Value
max_sigma = 20;

% Initialize (Solution) Vector
x_tr = zeros(size(x));

% Apply Truncated SVD
for i = 1:max_sigma
    x_tr = x_tr + (U(:,i).' * b) / S(i, i) * V(:,i);
end

% Display Values
% disp(reconstruct)

% Plot
figure;
plot(x_tr);

% Title
title('Truncated SVD');

%% Tikhonov (L2) Regularization
% Compute the L-Curve for the Tikhonov regularization problem with these
% variables. Use the maximum of the curvature plot to find a near-optimal
% value for 'λ'.

% Number of lambda samples to try.
num_lambda = 1000;

% Perturbation Errors
pert_error = zeros(num_lambda, 1);

% Bias Errors
bias_error = zeros(num_lambda, 1);

% Create a logspace of all of the lambdas to try.
lambda = logspace(-5, 2, num_lambda);

% Curvature
c_hat = zeros(num_lambda, 1);

% Apply Update Algorithm
for i = 1:num_lambda
    reconstruct = (A' * A + lambda(i)^2 * eye(N)) \ A' * b;
    pert_error(i) = norm(A * reconstruct - b, 2)^2;
    bias_error(i) = norm(reconstruct, 2)^2;
    
    z_lambda = (A' * A + lambda(i)^2 * eye(N)) \ A' * (A * reconstruct - b);
    bias_der = 4 / lambda(i) * reconstruct' * z_lambda;
    
    numerator = lambda(i)^2 * bias_der * pert_error(i) + 2 * lambda(i) * ...
        bias_error(i) * pert_error(i) + lambda(i)^4 * bias_error(i) * bias_der;
    denominator = (lambda(i)^2 * bias_error(i)^2 + pert_error(i)^2)^(3/2);
    c_hat(i) = 2 * bias_error(i) * pert_error(i) / bias_der * numerator / denominator;
end

%% Plot L-Curve

figure;

% Subplot
subplot(1, 2, 1);
plot(1 / 2 * log10(pert_error), 1 / 2 * log10(bias_error));

% Title
title('L Curve');

% Labels
xlabel('LOG Residual Norm');
ylabel('LOG Solution Norm');

% Subplot
subplot(1, 2, 2);
plot(log10(lambda), c_hat);

% Labels
xlabel('LOG Lambda');
ylabel('Curvature');

%% Optimal 'λ' Value

% For the optimal 'λ' value, just look at the curvature plot to see which 'λ' 
% produced the largest value.
[argvalue, argmax] = max(c_hat);
lambda = lambda(argmax);

% Display Values
disp(lambda);

%% Compute Tikhonov Solution

% Compute the Tikhonov solution 'x_{L2}' to this problem using the optimal 
% 'λ' found previously.
x_L2 = (A' * A + lambda^2 * eye(N)) \ A' * b;

% Plot
figure;
plot(x_L2);

% Title
title('Tikhonov/Ridge (L2) Solution');

%% L1 Regularization

% Create the sparsifying transform matrix 'L' using the following code.
L = -2 * eye(N) + diag(ones(N - 1, 1), 1) + diag(ones(N - 1, 1), -1);

% This matrix acts as a second derivative operator. Compute 'Lx' using the
% ground truth 'x' and plot the result.
value = L * x;

% Plot
figure;
plot(value);

% Title
title("2^{nd} Derivative Operator");

% We find that the result (displayed in the plot) represents a vector that
% is sparse in nature.

% This is the case as the the matrix 'L' acts as a 2nd derivative
% operator and 'x' is piecewise-linear. Thus, the 'spikes' displayed in the
% plot represent points where 'x' is discontinuous (i.e. the 2nd derivative
% is not equal to zero).

%% Compute L1 Solution
% Let 's' in R^n such that 'x = L^{-1} s'. Using 'λ = 0.0001', find the solution
% 's_{L1}' to the following L1 regularization problem:

% s_{L1} = min_{s} norm{A L^{-1} s - b}_2^2 + λ norm{s}_1

% Compute 'x_{L1} = L^{-1} s_{L1}', using the 'lasso' function in MATLAB.
% The value 's_{L1}' is sparse, and 'x_L1' is recovered from this.
x_L1 = L \ lasso(A / L, b, 'Lambda', 0.0001);

% Plot
figure;
plot(x_L1);

% Title
title('Lasso (L1) Solution');

%% Plots
figure;

plot(x, 'k');

hold on;
    plot(x_ls, 'r');
    plot(x_tr, 'b');
    plot(x_L2, 'g');
    plot(x_L1, 'm');
hold off;

lgd = legend('X', 'Least Squares', 'Truncated SVD', 'L2', 'L1');
fontsize(lgd, 8, 'Points');

%% Plots
figure;

plot(x, 'k');

hold on;
    plot(x_tr, 'b');
    plot(x_L2, 'g');
    plot(x_L1, 'm');
hold off;

ylim([-2 3]);

lgd = legend('X', 'Truncated SVD', 'L2', 'L1');
fontsize(lgd, 8, 'Points');

%% Save Results
filename = "model_1.mat";
save(filename);
