%% Assignment #2
% Carter Kruse, July 26th

% Each computational question will build on the previous. When MATLAB
% specific code is given, tools such as ChatGPT or other
% language-conversion programs may be used to translate the code into a
% different coding language.

% Let n = 100 and m = 80.
n = 100;
m = 80;

%% 1) Set-Up
% Generate a matrix 'A' in R^{m x n} such that 'A' is a matrix of normally 
% distributed random numbers with mean '0' and standard deviation '1'.

% In MATLAB, this can be accomplished using the 'randn' function.
A = randn(m, n);

% Generate a vector 'x' using the following code:
x = zeros(n, 1) ;
x(10:30) = 3 - 1/10 * (10:30);
x(30:60) = -3 + 1/10 * (30:60);
x(60:100) = -6 + 1/10 * (60:100);

% This should result in an 'x' that is piecewise-linear.

%% Computation

% Compute b = Ax + ε, where ε is Gaussian white noise with standard
% deviation η = 1.
eta = 1; epsilon = eta * randn(m, 1);
b = A * x + epsilon;

%% 2) L-Curve
% Compute the L-Curve for the Tikhonov regularization problem with these
% variables. Use the maximum of the curvature plot to find a near-optimal
% value for λ.

% Adapt the relevant sections of the Tikhonov example code on Canvas to
% compute the L-curve and curvature.

% Number of lambda samples to try.
num_lambda = 1000;

% Perturbation Errors
pert_error = zeros(num_lambda, 1);

% Bias Errors
bias_error = zeros(num_lambda, 1);

% Create a logspace of all of the lambdas to try.
lambda = logspace(-5, 2, num_lambda);

% Curvature
c_hat = zeros(num_lambda,1);

% Apply Update Algorithm
for i = 1:num_lambda
    fx_reconstruct = (A' * A + lambda(i)^2 * eye(n)) \ A' * b;
    pert_error(i) = norm(A * fx_reconstruct - b, 2)^2;
    bias_error(i) = norm(fx_reconstruct, 2)^2;
    
    z_lambda = (A' * A + lambda(i)^2 * eye(n)) \ A' * (A * fx_reconstruct - b);
    bias_der = 4 / lambda(i) * fx_reconstruct' * z_lambda;
    
    numerator = lambda(i)^2 * bias_der * pert_error(i) + 2 * lambda(i) * ...
        bias_error(i) * pert_error(i) + lambda(i)^4 * bias_error(i) * bias_der;
    denominator = (lambda(i)^2 * bias_error(i)^2 + pert_error(i)^2)^(3/2);
    c_hat(i) = 2 * bias_error(i) * pert_error(i) / bias_der * numerator / denominator;
end

%% Display Plots

% Display Figure
figure;

% Subplot
subplot(1, 2, 1);
plot(1/2 * log10(pert_error), 1/2 * log10(bias_error));

% Title/Labels
title('L Curve');
xlabel('LOG Residual Norm');ylabel('LOG Solution Norm');

% Subplot
subplot(1, 2, 2);
plot(log10(lambda), c_hat);

% Labels
xlabel('LOG Lambda'); ylabel('Curvature');

%% Optimal λ Value

% For the optimal λ value, just look at the curvature plot to see which λ 
% produced the largest value.
lambda = 10^0.68969; disp(lambda);

%% 3) Compute Tikhonov Solution

% Compute the Tikhonov solution 'x_{L2}' to this problem using the optimal 
% λ found previously.
x_L2 = (A' * A + lambda^2 * eye(n)) \ A' * b;

%% 4) Matrix L

% Form a matrix 'L' using the following code:
L = -2 * eye(n) + diag(ones(n - 1, 1), 1) + diag(ones(n - 1, 1), -1);

% This matrix acts as a second derivative operator. Compute 'Lx' using the
% ground truth 'x' and plot the result.
value = L * x;
figure; plot(value);

%% Explanation

% We find that the result (displayed in the plot) represents a vector that
% is sparse in nature.

% This is the case as the the matrix 'L' acts as a 2nd derivative
% operator and 'x' is piecewise-linear. Thus, the 'spikes' displayed in the
% plot represent points where 'x' is discontinuous (i.e. the 2nd derivative
% is not equal to zero).

%% 5) L1 Regularization
% Let 's' in R^n such that x = L^{-1} s. Using λ = 0.01, find the solution
% 's_{L1}' to the following L1 regularization problem:

% s_{L1} = min_{s} norm{A L^{-1} s - b}_2^2 + λ norm{s}_1

% Compute x_{L1} = L^{-1} s_{L1}.
% Hint: Use the 'lasso' function in MATLAB.
x_L1 = L \ lasso(A / L, b, 'Lambda', 0.01);

%% 6) Display Plots
% Plot 'x_L2' and 'x_L1' and compare these results.

% Display Figure
figure;

% Plots
hold on;
    plot(x, 'k');
    plot(x_L2, 'r');
    plot(x_L1, 'b');
hold off;

% Legend
legend('X', 'L2 (Tikhonov)', 'L1');

%% Explanation

% In comparing the results, we find that the L1 solution is better than the
% L2 (Tikhonov) solution, as it matches the actual 'x' better.

% This highlights the use of the L1 regularization compared to the L2
% regularization in specific cases.
