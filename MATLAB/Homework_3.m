%% Assignment #3
% Carter Kruse, August 9th

%% Question 1
% Generate random samples of Z and plot them in a histogram.

clearvars;
samples = 10000;

X = rand(samples, 1);
Z = (2 .* (X.^3)) + 3;

% Display Figure
figure;

% Histogram
histogram(Z);

% Title
title('Histogram Of Z = 2X^3 + 3');

% Labels
xlabel('Z Values'); ylabel('Frequency');

%%
% Plot the probability density function you calculated in (a).

z = linspace(3, 5);
probability = (1 / 6) .* ((1 / 2) .* (z - 3)).^(-2 / 3);

% Display Figure
figure;

% PDF
plot(z, probability);

% Title
title('PDF');

% Labels
xlabel('Z Values'); ylabel('Density');

%%
% Compare these results.

% Display Figure
figure;

% Plot (Overlaid)
hold on;
    histogram(Z, "Normalization", "pdf");
    plot(z, probability, 'r');
hold off;

% Title
title('Comparison');

% Labels
xlabel('Z Values'); ylabel('Density');

% The results are equivalent, indicating that the probability density
% function for the random variable Z is appropriate.

%% Question 2
% Given alpha = 2 and beta = 1 / 2, plot π(y|x).

clearvars;
alpha = 2;
beta = 1 / 2;

y = linspace(0, 20);

%% x = 1 / 10

x = 1 / 10;

likelihood = (1 / (2 .* x.^2 + 1)) .* (((beta.^alpha) / (gamma(alpha))) .* ...
    (y / (2 .* x.^2 + 1)).^(alpha - 1) .* exp(-beta .* (y / (2 .* x.^2 + 1))));

% Display Figure
figure;

% Plot
plot(likelihood);

% Title
title('Likelihood (x = 1 / 10)');

%% x = 1 / 2

x = 1 / 2;

likelihood = (1 / (2 .* x.^2 + 1)) .* (((beta.^alpha) / (gamma(alpha))) .* ...
    (y / (2 .* x.^2 + 1)).^(alpha - 1) .* exp(-beta .* (y / (2 .* x.^2 + 1))));

% Display Figure
figure;

% Plot
plot(likelihood);

% Title
title('Likelihood (x = 1 / 2)');

%% x = 1

x = 1;

likelihood = (1 / (2 .* x.^2 + 1)) .* (((beta.^alpha) / (gamma(alpha))) .* ...
    (y / (2 .* x.^2 + 1)).^(alpha - 1) .* exp(-beta .* (y / (2 .* x.^2 + 1))));

% Display Figure
figure;

% Plot
plot(likelihood);

% Title
title('Likelihood (x = 1)');

%% Question 3
% Write code that generates random samples of X for alpha = 1.

clearvars;

% Parameters

alpha = 1;
samples = 1000;

% Generate random samples from uniform distribution.
U = rand(samples, 1);

% Transform the distribution, as appropriate.
X = (1 / alpha) .* tan(pi .* (U - (1 / 2)));

% Display Figure
figure;

% Histogram
histogram(X, 3000, "Normalization", "pdf");
xlim([-15 15]);

% Title
title('Random Samples (Cauchy Density)');

% Labels
xlabel('X'); ylabel('Probability Density');

%% Question 4
% Write code to generate random samples from π_pr(x) when N = 100 and alpha = 1.

clearvars;

% Parameters
N = 100;
alpha = 1;

xi = 1 / alpha * tan(pi * rand(N - 1, 1) - pi / 2);
cauchy_vector = [0; cumsum(xi)];

% Set initial values to 0.
x(1:20) = 0;

% Recursive (Update Loop)
for jj = 20:N-1
    x(jj + 1) = xi(jj) + 2 * x(jj) - x(jj - 1);
end

% Display Figure
figure();

% Plots
plot(x);
% plot(diff(x));
% plot(diff(diff(x)));

% plot(xi);
% plot(cauchy_vector);
