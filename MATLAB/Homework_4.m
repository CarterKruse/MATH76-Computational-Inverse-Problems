%% Assignment #4
% Carter Kruse, August 23rd

%% Question 1
% Generate samples from the posterior density function and plot your
% results.

clearvars;

% Conditional mean of the multivariate Gaussian.
mu = [1.7353; 0.0319];

% Conditional variance matrix of the multivariate Gaussian.
covariance = [0.2647 -0.0319; -0.0319 0.0239];

% Number of points (vectors) to generate from the posterior density
% function.
num_samples = 5000;

% The following function returns a matrix of 'n' random vectors chosen from
% the same multivariate normal distribution, with mean vector 'mu' and
% covariance matrix sigma.
R = mvnrnd(mu, covariance, num_samples);

% Display figure.
figure;

% Plot the samples.
plot(R(:, 1), R(:, 2), '.b');
% scatter(R(:, 1), R(:, 2), '.');

% Title
title('Posterior Density Function');

% Labels
xlabel('X');
ylabel('Y');

%% Question 2
% Generate samples from π and plot your results.
% Metropolis-Hastings Algorithm

clearvars;

% Length of Markov chain.
N_M = 5000;

% Variance of Gaussian proposal density.
gamma = 0.8;

% Density to sample from.
prob = @(x) exp(-abs(x(1,:).^2 + x(2,:).^2 - 5));

% Physical grid for plotting contour curves.
[X, Y] = meshgrid(linspace(-3, 3, 1000), linspace(-3, 3, 1000));

% Density value on physical grid.
Z = prob([X(:)'; Y(:)']);

% Plot contour curves.
figure(1);
contour(X, Y, reshape(Z, 1000, 1000)); hold on;

% Proposal density.
q = @(x, y) exp(-1 / (2 * gamma^2) * norm(x - y, 2).^2);

% Initialize Markov chain.
x = zeros(2, N_M);

% Set first entry of Markov chain.
x(:, 1) = [3; -1];

% Perform Metropolis-Hastings algorithm.
for kk = 2:N_M
    y = x(:, kk - 1) + gamma * randn(2, 1);
    alpha = min(1, prob(y) / prob(x(:, kk - 1)));
    if rand < alpha
        x(:, kk) = y;
    else
        x(:, kk) = x(:, kk - 1);
    end

    % Plot samples as you go.
    % figure(1);
    % plot(x(1, 1:kk), x(2, 1:kk), '.b'); pause(0.1);
end

% Plot samples.
figure(1);
plot(x(1, :), x(2, :), '.b'); pause(0.01); hold off;

% Plot trace plot of x_1.
figure(2);
plot(x(1, :), 'b');

%% Question 3
% Generates samples of π using a Gibbs sampling technique. Plot your
% results.

clearvars;

% Points are sampled from joint probability density function, using
% conditional probabilities.

% Number of samples.
num_samples = 5000;

% Parameters of the joint Gaussian distribution.
mu = [3; 2];
covariance = [4 2; 2 3];

% Initialize samples array/matrix.
samples = zeros(num_samples, 2);

% Initialize variables.
x = 0;
y = 0;

% Gibbs Sampling
for i = 1:num_samples
    % Sample 'x' from its conditional distribution (given 'y').
    cov_x_given_y = covariance(1, 1) - covariance(1, 2) * (1 / covariance(2, 2)) * covariance(2, 1);
    mu_x_given_y = mu(1) + covariance(1, 2) * (1 / covariance(2, 2)) * (y - mu(2));
    x = mu_x_given_y + sqrt(cov_x_given_y) * randn();
    
    % Sample 'y' from its conditional distribution (given 'x').
    cov_y_given_x = covariance(2, 2) - covariance(2, 1) * (1 / covariance(1, 1)) * covariance(1, 2);
    mu_y_given_x = mu(2) + covariance(2, 1) * (1 / covariance(1, 1)) * (x - mu(1));
    y = mu_y_given_x + sqrt(cov_y_given_x) * randn();
    
    samples(i, :) = [x, y];
    % figure; plot(x, y, '.b'); pause(0.01);
end

% Display figure.
figure;

% Plot the samples.
plot(samples(:, 1), samples(:, 2), '.b');
% scatter(samples(:, 1), samples(:, 2), '.b');

% Title
title('Joint Density Function (Gibbs Sampling)');

% Labels
xlabel('X');
ylabel('Y');
