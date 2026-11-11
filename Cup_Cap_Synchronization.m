function Cup_Cap_Synchronization(M, eta, h1, h2, sigma1_range, sigma2, sigma3, dist, Tmax, Tmin, model)
%Goal: Illustrate the synchronization and desynchronization process, and measured by order parameters.
%General Setting: N = 10000 nodes, RK4 with dt = 0.005
%                 Sweeping sigma1 with sigma2 (and sigma3) fixed at step dsigma = 0.005
%=========================================Input:=========================================
%   M: maximum order for the model. 2 (up to triangles) or 3 (up to tetrahedrons).
%   eta: Bias, 0 ≤ eta ≤　1. eta and 1 - eta are expected to have the same behaviour.
%   h1: f1(x) = x^h1.
%   h2: f2(x) = x^h2. It is zero if M = 2.
%   sigma1_range: List [sigma1_min, sigma1_max]. The range to sweep sigma1.
%   sigma2: Coupling constant for triangles.
%   sigma3: Coupling constant for tetrahedrons. It is zero if M = 2.
%   dist: The distribution g(omega). Gaussian or Lorentzian
%   Tmax: default 20. 
%   Teq: default 15. The equilibrium threshold for RK4. Taking average over [Teq, Tmax]  
%   model: defines theta0 when it's biased. 1 (R) or 2 (X_half).
% ===============================Settings by cases=======================================
% M = 2, h1 odd: theta0 = 2*pi*rand.
%                Perfrom synchronization (increasing sigma1) first, then desynchronize (decreasing sigma1).
% M = 2, h1 even: model 1: theta0 = (pi*r + pi/2).*(x < eta) + (pi*r + 5*pi/2).*(x >= eta)
%                 model 2: theta0 = pi*(x < eta) + 3*pi*(x >= eta)
%                 Perfrom desynchronization with biased initial phases. Perform synchronization with bias = 1/2.
% M = 3: theta0 = (pi*r + pi/2).*(x < eta) + (pi*r + 5*pi/2).*(x >= eta)
%        Perfrom desynchronization with biased initial phases. Perform synchronization with bias = 1/2.
% ==============================Important formulas:======================================
% R: order parameters, average over [Teq, Tmax] for |sum(exp(1i*theta))/N|
% X_{1/2}: half-integer order parameters, average over [Teq, Tmax] for sum(exp(1i*theta/2))/N
% =======================================Output:=========================================
% <name>.mat: sup, Rup, sdown, Rdown, theta_final_up, theta_final_down, params (includes. params.seed); 
%             model 2 adds Xhalf_down
% <name>.fig.png: Plot R against sigma1, synchronization (red) and desynchronization (blue)
% <name>_Xhalf.fig.png  model 2 only: Plot Im <X_{1/2}> against sigma1
% =====================================Defaults:=========================================
% When M = 2, h2 and sigma3 are set to be 0.
% The biased initial phases are assued to have distribution, i.e., model 1.
% Assuming the Gaussin distribution.

if nargin < 4  || isempty(h2),     h2     = 0;              end
if nargin < 7  || isempty(sigma3), sigma3 = 0;              end
if nargin < 8  || isempty(dist),   dist   = 'Gaussian';     end
if nargin < 9  || isempty(Tmax),   Tmax   = 20;             end
if nargin < 10  || isempty(Tmin),   Tmin   = 15;             end
if nargin < 11 || isempty(model),  model  = 1;              end
params = initial_parameters(M, eta, h1, h2, sigma1_range, sigma2, sigma3, dist, Tmax, Tmin, model);

% synchronization: sweep sigma1 up
sup = params.sigma_min:params.ds:params.sigma_max;
[Rup, Xhalf_up, theta_final_up] = sweep(params.theta0, sup, 'sync', params);

% desynchronization: sweep sigma1 down
% M = 2 with h1 odd: using the theta_final_up as the startof theta_0.
% Otherwise, theta0 back to initial
if strcmp(params.type, 'M2_hodd')
    theta_start = theta_final_up;
else
    theta_start = params.theta0;
end
sdown = params.sigma_max:-params.ds:params.sigma_min;
[Rdown, Xhalf_down, theta_final_down] = sweep(theta_start, sdown, 'desync', params);

%Save data
name   = make_name(params);
theta0 = params.theta0;
vars   = {'sup', 'Rup', 'sdown', 'Rdown', 'theta_final_up', 'theta_final_down', 'params'};
if params.Xhalf
    vars = [vars, {'Xhalf_up', 'Xhalf_down'}];
end
save([name '.mat'], vars{:});
fprintf('Saved: %s.mat\n', name);

%Create figure for Order parameters
fig = figure();
plot_simulation(sup, Rup, sdown, Rdown, 'R_\theta', '-', params);
finish_figure(fig, 'R_{\theta}', [0 1], params, name);
if params.Xhalf
    fig = figure();
    plot_imaginary(sdown, imag(Xhalf_down), 'Im X_{1/2}', '-', params);
    finish_figure(fig, 'Im X_{1/2}', [-1 1], params, [name '_Xhalf']);
end
end

% =============================================================================================
% Initial settings
% =============================================================================================
function params = initial_parameters(M, eta, h1, h2, sigma1_range, sigma2, sigma3, dist, Tmax, Teq, model)
    params.N      = 10000;
    params.M      = M;
    params.eta    = eta;
    params.h1     = h1;
    params.h2     = h2;
    params.sigma2 = sigma2;
    params.sigma3 = sigma3;
    params.dist   = dist;
    params.model  = model;
    params.ds     = 0.005;
    params.dt     = 0.005;              
    params.Teq  =  Teq;
    params.Tmax = Tmax;
    params.sigma_min = sigma1_range(1);
    params.sigma_max = sigma1_range(2);
    rng('shuffle');            % seed from the current time
    s = rng;
    params.seed = s.Seed; 
    if M == 3
        params.type = 'M3';
    elseif mod(h1, 2) == 1
        params.type = 'M2_hodd';
    else
        params.type = 'M2_heven';
    end
    model2      = strcmp(params.type, 'M2_heven') && model == 2;
    params.Xhalf= model2;

    % initial frequency w based on Distribution g(w).
    N = params.N;
    if strcmp(dist, 'Lorentzian')
        params.w = tan(pi*(rand(N,1) - 0.5));
    else
        params.w =  randn(N,1);
    end

    % initial phase theta0
    x = rand(N, 1);
    if strcmp(params.type, 'M2_hodd')
        params.theta0 = 2*pi*x;
    else
        r = rand(N, 1);                 
        if model2
            params.theta0 = pi*(x < eta) + 3*pi*(x >= eta);
        else
            params.theta0 = (pi*r + pi/2).*(x < eta) + (pi*r + 5*pi/2).*(x >= eta);
        end
    end
end
% =============================================================================================
% Main code for the Cup Cap Synchronization Model
% =============================================================================================
function theta_dot = CupCapDynamics(theta, sigma1, params)
% Input:
%   theta: phase, N x 1
%   sigma1: float in sigma1_range, coupling constant
%   params: N, M, h1, h2, sigma2, sigma3, w
% Output:
%   theta_dot: dynamics on the nodes, which is made up with
%       theta_dot_i = w_i + Im( Lambda_i * exp(-1i*theta_i) ),
%       Lambda_i    = sigma1*X_1 + sigma2*Y_i + sigma3*Z_i (Z_i for M = 3 only)

    N = params.N;
    h1 = params.h1;
    h2 = params.h2;
    E = exp(1i*theta);
    X1 = mean(E);
    
% sigma2 term:
    Y = zeros(N,1);
    for n = 0:h1
        n1 = (2*n - h1 + 1)/2;
        n2 = (-2*n + h1 + 1)/2;
        E1 = exp(1i * n1 * theta);
        E2 = exp(1i * n2 * theta);
        Xn1 = (sum(E1) - E1)/N;
        Xn2 = (sum(E2) - E2)/N;
        Y = Y + nchoosek(h1, n)*(Xn1.*Xn2 - (sum(E) - E)/N^2)/2^h1;
    end
% sigma3 term (if M = 3):
    Z = zeros(N,1);
    if params.M == 3
        for n = 0:h1-1
            for m = 0:h2-1
                n1 = (1 - 2*n + h1)/2;
                n2 = (2*n - 2*m - h1 + h2)/2;
                n3 = (2*m - h2 + 1)/2;
                E1 = exp(1i * n1 * theta);
                E2 = exp(1i * n2 * theta);
                E3 = exp(1i * n3 * theta);
                E12 = exp(1i * (n1+n2) * theta);
                E13 = exp(1i * (n1+n3) * theta);
                E23 = exp(1i * (n2+n3) * theta);
                Xn1 = (sum(E1) - E1)/N;
                Xn2 = (sum(E2) - E2)/N;
                Xn3 = (sum(E3) - E3)/N;
                A12  = (sum(E12) - E12)/N^2;
                A13  = (sum(E13) - E13)/N^2;
                A23  = (sum(E23) - E23)/N^2;
                A123 = (sum(E) - E)/N^3;
                Z = Z + nchoosek(h1-1, n)*nchoosek(h2-1, m)*(Xn1.*Xn2.*Xn3 - A12.*Xn3 - A13.*Xn2 - A23.*Xn1 + 2*A123)/2^(h1+h2-2);
            end
        end
    end
    Lambda    = sigma1*X1 + params.sigma2*Y + params.sigma3*Z;
    theta_dot = params.w + imag(Lambda.*conj(E));  
end
% =============================================================================================
% RK4 for intergrate the dynamic
% =============================================================================================
function theta = rk4_step(theta, sigma1, params)
    dt = params.dt;
    k1 = CupCapDynamics(theta,sigma1, params);
    k2 = CupCapDynamics(theta + 0.5*dt*k1,sigma1, params);
    k3 = CupCapDynamics(theta + 0.5*dt*k2,sigma1, params);
    k4 = CupCapDynamics(theta + dt*k3, sigma1, params);
    theta = theta + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
end
% =============================================================================================
% Intergrate the dynamic
% =============================================================================================
function [R, Xhalf, theta] = integrate_dynamics(theta, sigma1, params)
% Input:
%   theta: phase, N x 1
%   sigma1: float in sigma1_range, coupling constant
%   params: N, M, h1, h2, sigma2, sigma3, w
% Output:
%   R: Order parameter (float)
%   Xhalf: half integer order parameter (complex)
%   theta: new phase, N x 1
    N = params.N;
    nmax = round(params.Tmax/params.dt);
    nmin = round(params.Teq/params.dt);
    R = 0;
    Xhalf = 0;
    if params.Xhalf
        for i = 1:nmax
            theta = rk4_step(theta, sigma1, params);
            if i > nmin
                R = R + abs(sum(exp(1i*theta)))/N;
                Xhalf = Xhalf + sum(exp(0.5i*theta))/N;
            end
        end
    else
        for i = 1:nmax
            theta = rk4_step(theta, sigma1, params);
            if i > nmin
                R = R + abs(sum(exp(1i*theta)))/N;
            end
        end
    end
    R = R/(nmax - nmin);
    Xhalf = Xhalf/(nmax - nmin);
end
% =============================================================================================
% Synchronization and Desynchronization by sweeping sigma1 with sigma2 (and sigma3) fixed
% =============================================================================================
function [R, Xhalf, theta] = sweep(theta, s, label, params)
% Input:
%   theta: phase, N x 1
%   s: sup: list sigma1_min:ds:sigma1_max; or sdown: list sigma1_max:ds:sigma1_min
%   label: "desync" or "sync"
%   params: N, M, h1, h2, sigma2, sigma3, w
% Output:
%   R: Order parameter (list)
%   Xhalf: half integer order parameter (list)
%   theta: final phase, N x 1
    ns = numel(s);
    R = zeros(1, ns);
    Xhalf = complex(zeros(1, ns));
    for n = 1:ns
        [R(n), Xhalf(n), theta] = integrate_dynamics(theta, s(n), params);
        if mod(n - 1, 1000) == 0 || n == ns
            fprintf('%s sigma1 = %+.3f; R = %.4f\n', label, s(n), R(n))
        end
    end
end
% =============================================================================================
% Generate names to record the data
% =============================================================================================
function name = make_name(params)
% File name without folder and extension; lists the parameters the case uses.
    p = params;
    T = sprintf('Teq=%g_Tmax=%g', p.Teq, p.Tmax);
    switch p.type
        case 'M2_hodd'
            name = sprintf('M=2_N=%d_sigma2=%.2f_h1=%d_%s_%s', p.N, p.sigma2, p.h1, p.dist, T);
        case 'M2_heven'
            name = sprintf('M=2_N=%d_sigma2=%.2f_h1=%d_%s_eta=%.2f_model=%d_%s', p.N, p.sigma2, p.h1, p.dist, p.eta, p.model, T);
        case 'M3'
            name = sprintf('M=3_N=%d_sigma2=%.2f_sigma3=%.2f_h1=%d_h2=%d_%s_eta=%.2f_%s', p.N, p.sigma2, p.sigma3, p.h1, p.h2, p.dist, p.eta, T);
    end
end

% =============================================================================================
% Plot the figure
% =============================================================================================
function plot_simulation(sup, yup, sdown, ydown, label, style, params)
    Marks = max(1, round(0.1/params.ds));
    plot(sup, yup, style, 'color', 'r', 'linewidth', 1, 'marker', 'o', 'markersize', 5, 'MarkerIndices', 1:Marks:numel(sup), 'displayname', [label ' sync']); hold on;
    plot(sdown, ydown, style, 'color', 'b', 'linewidth', 1, 'marker', 'd', 'markersize', 5, 'MarkerIndices', 1:Marks:numel(sdown), 'displayname', [label ' desync']);
end
 
function plot_imaginary(sdown, ydown, label, style, params)
    Marks = max(1, round(0.1/params.ds));
    plot(sdown, ydown, style, 'color', 'b', 'linewidth', 1, 'marker', 'd', 'markersize', 5, 'MarkerIndices', 1:Marks:numel(sdown), 'displayname', [label ' desync']);
end
% =============================================================================================
% Settings for the figure
% =============================================================================================
function finish_figure(fig, ylab, ylims, params, file)
    set(gca, 'Xlim', [params.sigma_min, params.sigma_max], 'Ylim', ylims)
    yticks(ylims(1):0.2:ylims(2))
    ytickformat('%.1f')
    xlabel('\sigma_1', 'Interpreter', 'tex', 'fontsize', 24, 'fontweight', 'bold')
    ylabel(ylab, 'Interpreter', 'tex', 'fontsize', 24, 'fontweight', 'bold')
    legend('location', 'best', 'Interpreter', 'tex', 'box', 'off', 'fontsize', 16, 'fontweight', 'bold')
    savefig(fig, [file '.fig']);
    saveas(fig, [file '.png']);
end