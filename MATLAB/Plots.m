%% Binary Signal Recovery
% Computational Inverse Problems, Math 076 (Summer 2023)
% Carter Kruse

clearvars;

%% Model 1

load('model_1.mat');

%% Mapping

x_tr_final = zeros(N, 1);
x_tr_final(1:N) = x_tr(1:N);
x_tr_final(x_tr_final < 0.5) = 0;
x_tr_final(x_tr_final > 0.5) = 1;

x_L2_final = zeros(N, 1);
x_L2_final(1:N) = x_L1(1:N);
x_L2_final(x_L2_final < 0.5) = 0;
x_L2_final(x_L2_final > 0.5) = 1;

x_L1_final = zeros(N, 1);
x_L1_final(1:N) = x_L1(1:N);
x_L1_final(x_L1_final < 0.5) = 0;
x_L1_final(x_L1_final > 0.5) = 1;

%% Plots

figure;
tiledlayout(2, 1);

nexttile;
plot(x, 'k');
hold on;
    plot(x_tr_final, 'b');
hold off;

ylim([-0.5 1.5]);

nexttile;
plot(x, 'k');
hold on;
    plot(x_L1_final, 'm');
hold off;

ylim([-0.5 1.5]);
