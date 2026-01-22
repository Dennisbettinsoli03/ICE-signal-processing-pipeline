clc;       
clear;     
close all; 
%% import data
file_name='datasheet_2025.xlsx';
sheet_name = 'Steady-state tests';
T=readtable(file_name, 'Sheet', sheet_name,'Range', 'A:J');
file_name='datasheet_2025.xlsx';
sheet_name = 'Engine data';
E=readtable(file_name, 'Sheet', sheet_name);
%% Initial parameters kPa and K
p0d=99;
T0d=298;
Q_lhv = 42.5; % MJ/kg
B = 95.8; % mm
S= 104; % mm
R_air = 287.05; % J/kgK
cp = 1038; % J/kgK
gamma = 1.4;
R_EGR = cp * (1 - 1/gamma);
i = 4;
% Calculate the displacement volume
V_d = (pi/4) * (B^2) * S;

% convert pressure from mbar to kPa
T.p_baro=T.p_baro./10;
T.p_i_MF=T.p_i_MF./10+101.325;
% convert temperature in K
T.T_snorkel = T.T_snorkel +273.15 ;
T.T_i_MF = T.T_i_MF +273.15;
% convert Q_lhv in kwh/kg
Q_lhv = Q_lhv * 0.2778;

%% Corrected power and torque (in kW and Nm)
T.qc=T.qm_fuel./T.p_i_MF.*T.p_baro;
T.fm= zeros(height(T),1);
T.fm(T.qc<37.2) = 0.2;
T.fm(T.qc >= 37.2 & T.qc<65) = 0.036 .*T.qc (T.qc>=37.2 & T.qc<= 65) - 1.14;
T.fm (T.qc>65)= 1.2;
T.fa= (p0d./T.p_baro).^0.7.*((T.T_snorkel)./T0d).^1.2;
T.mu_c = T.fa.^T.fm;
T.P = T.T_dyno .* 2 .* pi .* T.n_engine./60 /1000;
T.P_0 = T.P .* T.mu_c;
T.T_0 = T.T_dyno .* T.mu_c;

%% Fuel conversion efficiency 
T.efficiency = T.P_0 ./ (Q_lhv * T.qm_fuel);

%% Brake specific fuel consumption in g/kWh
T.bsfc = 1 ./ (T.efficiency.* Q_lhv) .*1000; 


%% Volumetric efficiency
T.m_air = T.qm_air *60 * 2 ./ (3600* i* T.n_engine); 
T.m_EGR = T.qm_EGR *60 * 2 ./ (3600* i* T.n_engine);
T.m_fuel= T.qm_fuel *60 * 2 ./ (3600* i* T.n_engine);
T.R_mix = (T.qm_air.*R_air+ T.qm_EGR* R_EGR)./(T.qm_air+T.qm_EGR); % J/kgK
T.vol_eff = ((T.m_air+T.m_EGR)./V_d).*((T.R_mix.*T.T_i_MF)./T.p_i_MF);
T.vol_eff = T.vol_eff* 10^6;

%% Plot Full load parameters
figure
plot(T.n_engine, T.P, '-o', 'Color','r', 'LineWidth',1, 'MarkerSize',6); hold on;
plot(T.n_engine, T.P_0, '-o', 'Color','g', 'LineWidth',1, 'MarkerSize',6);
plot(T.n_engine, T.T_dyno, '-o', 'Color','b', 'LineWidth',1, 'MarkerSize',6);
plot(T.n_engine, T.T_0, '-o', 'Color','y', 'LineWidth',1, 'MarkerSize',6);
grid on
grid minor
box on
ylim([0,500])
xlabel('Engine speed (RPM)', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Full load characteristics', 'FontSize', 14, 'FontWeight', 'bold')
title('Full load characteristics over engine speed', 'FontSize', 20)
set(gca, 'FontSize', 12)
legend({'P_{dyno}','P_0','T_{dyno}','T_0'}, 'Location','best', 'FontSize',12);

%% Plot correction factor
figure
plot(T.n_engine, T.mu_c, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'Color', [0 0.4470 0.7410])
grid on
grid minor
box on
ylim([0.99, 1.005])
xlabel('Engine speed (RPM)', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Correction factor', 'FontSize', 14, 'FontWeight', 'bold')
title('Correction factor over engine speed', 'FontSize', 20)
set(gca, 'FontSize', 12)

%% Plot fce
figure
plot(T.n_engine, T.efficiency, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'Color', [0 0.4470 0.7410])
grid on
grid minor
box on
ylim([0.30, 0.45])
xlabel('Engine speed (RPM)', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Fuel conversion efficiency', 'FontSize', 14, 'FontWeight', 'bold')
title('Fuel conversion efficiency over engine speed', 'FontSize', 20)
set(gca, 'FontSize', 12)

%% Plot BSFC
figure
plot(T.n_engine, T.bsfc, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'Color', [0 0.4470 0.7410])
grid on
grid minor
box on
ylim([200, 245])
xlabel('Engine speed (RPM)', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Brake specific fuel consumption', 'FontSize', 14, 'FontWeight', 'bold')
title('Brake specific fuel consumption over engine speed', 'FontSize', 20)
set(gca, 'FontSize', 12)

%% Plot volumetric efficiency
figure
plot(T.n_engine, T.vol_eff, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'Color', [0 0.4470 0.7410])
grid on
grid minor
box on
ylim([0,1])
xlabel('Engine speed (RPM)', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Volumetric efficiency', 'FontSize', 14, 'FontWeight', 'bold')
title('Volumetric efficiency over engine speed', 'FontSize', 20)
set(gca, 'FontSize', 12)

%% Load data
load('ifile_2000FL.mat'); 
%% Crank angle theta
theta_deg = linspace(0,719.9,7200);

%% Cylinder pressure: compute mean over first 100 cycles
p_raw  = double(ifile.PCYL1.data);
p_mean = mean(p_raw(:, 1:100), 2);   

%% Manifold pressure: compute mean over first 100 cycles
p_man       = double(ifile.PMAN1.data);
p_man_mean  = mean(p_man(:, 1:100), 2);                 

%% Plot cylinder mean pressure
figure
plot(theta_deg, p_mean, 'LineWidth', 2)
grid on; grid minor; box on
xlabel('Crank angle', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Mean pressure', 'FontSize', 14, 'FontWeight', 'bold')
title('Mean cylinder pressure over crank angle', 'FontSize', 20)
set(gca, 'FontSize', 12)
legend({'P_{cyl}'}, 'Location','best', 'FontSize',12);


%% Compute pegged signal 
ifile.PMAN1.axis(:,2) = p_mean;

% Interval for reference (175–185 deg)
idx = (theta_deg > 175) & (theta_deg < 185);
p_mean_interval  = mean(ifile.PMAN1.axis(idx, 2));

% Global manifold mean
p_mean_manifold  = mean(p_man_mean);

% Offset
p_offset = p_mean_manifold - p_mean_interval;

% Pegged pressure
p_pegged = (+p_mean + p_offset);   % segno coerente

figure
plot(theta_deg, p_pegged, 'LineWidth', 2)
grid on; grid minor; box on
title('Pegged pressure')
xlabel('Crank angle')
ylabel('Pressure')
xlim([280 450])


%% FIR moving average filter
windowSize = 10;
b = (1/windowSize) * ones(1, windowSize);
a = 1;
p_filtered = filter(b, a, p_pegged);

%% Butterworth filter
enc_res = 0.1;
fs = 2000 ./ 60 .* 360 ./ enc_res;   % sampling frequency

fc = 4000;            % cutoff [Hz]
Wn = fc / (fs/2);     % normalized cutoff
n  = 2;               % order

[b,a] = butter(n, Wn);

p_cyl_butter_filtfilt = filtfilt(b, a, p_pegged);    % zero-phase
p_cyl_butter_filt     = filter(b, a, p_pegged);      % causal

%% moving mean
p_moving = movmean(p_pegged, 10);

%% Plot comparison
figure(100)
plot(theta_deg, p_pegged,                'LineWidth', 2); hold on
plot(theta_deg, p_filtered,              'LineWidth', 2);
plot(theta_deg, p_cyl_butter_filtfilt,       'LineWidth', 2);
plot(theta_deg, p_moving,                'LineWidth', 2);

grid on; grid minor; box on
xlabel('Crank angle', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Pressure',      'FontSize', 14, 'FontWeight', 'bold')
title('Pressure filtering comparison', 'FontSize', 20)
set(gca, 'FontSize', 12)
xlim([366 378])
ylim([135 140])
legend({'Input Data', 'FIR Moving Avg', 'Butterworth', 'movmean'}, ...
       'Location','northeast', 'FontSize',12)

%% Volume calculation
A_piston = pi * (ifile.engine.bore)^2 /4;      % mm^2
r = 0.5 * ifile.engine.stroke;
l = ifile.engine.conrod_length;
theta = deg2rad(theta_deg);
lambda = r/l;
x = r.*((1 - cos(theta)) + 1./lambda*(1 - sqrt(1 - (lambda^2 .* sin(theta).^2))));
V_d = A_piston * ifile.engine.stroke * 10^-3 * i;
V_c = V_d / (ifile.engine.compression_ratio  -1)/i;  % Assuming V_c is the clearance volume
V_x = V_c + A_piston *10^(-3) .* x;


%% Plots
% Plot p-theta and p-V
figure
plot (theta_deg, V_x,  'LineWidth', 2)
grid on; grid minor; box on
xlabel('Crank angle', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Volume [cm^3]',      'FontSize', 14, 'FontWeight', 'bold')
title('Volume over crank angle', 'FontSize', 20)
set(gca, 'FontSize', 12)
figure
plot (theta_deg, p_cyl_butter_filtfilt,  'LineWidth', 2)
grid on; grid minor; box on
xlabel('Crank angle', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Volume',      'FontSize', 14, 'FontWeight', 'bold')
title('Pressure over crank angle', 'FontSize', 20)
set(gca, 'FontSize', 12)

figure
plot (V_x, p_cyl_butter_filtfilt,  'LineWidth', 2)
grid on; grid minor; box on
xlabel('Volume', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Pressure',      'FontSize', 14, 'FontWeight', 'bold')
title('Gas pressure', 'FontSize', 20)
set(gca, 'FontSize', 12)

V_ratio = V_x ./ max(V_x);
figure
plot (V_ratio, p_cyl_butter_filtfilt,  'LineWidth', 2)
grid on; grid minor; box on
xlabel('Volume [l]', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Pressure [bar]',      'FontSize', 14, 'FontWeight', 'bold')
title('P-V diagram', 'FontSize', 20)
set(gca, 'FontSize', 12)


figure
loglog(V_ratio,p_cyl_butter_filtfilt, 'LineWidth', 2)
grid on; grid minor; box on
xlabel('Volume ratio', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Pressure [bar]', 'FontSize', 14, 'FontWeight', 'bold')
title('P-V diagram', 'FontSize', 20)
set(gca, 'FontSize', 12)


% calculate Bmep - Imep (bar)
% values added on V_x and p_pegged for computation of imep_net and gross
p_pegged = p_pegged';
V_x = [V_x V_x(1)];
p_pegged = [p_pegged p_pegged(1)];
A=size(V_x);
B=size(p_pegged);
%-----------------------------------------------------------------------------%
w_i = trapz(V_x,p_pegged);
imep_n =i* w_i / V_d;
theta_g =(theta_deg(:,1801:5401));
theta_g_rad=deg2rad(theta_g);
x_gross = r.*((1 - cos(theta_g_rad)) + 1./lambda*(1 - sqrt(1 - (lambda^2 .* sin(theta_g_rad).^2))));
V_gross = V_c + A_piston .* x_gross *10^-3;
p_gross = p_pegged(:,1801:5401);      %id);
w_g = trapz(V_gross, p_gross);
imep_g = w_g / V_d*4;
bmep = 4*pi * T.T_0(7) /(V_d)* 10;
mec_eff = bmep/imep_g;
fprintf ('imep gross = %f\n',imep_g)
fprintf ('imep net = %f\n',imep_n)
fprintf ('Bmep = %f\n',bmep)
fprintf ('Mechanical efficiency = %f',mec_eff)

%% Part 2: FOURIER ANALYSIS

% downsampling from 7200 to 720
DSF = 10; 
p_DSF=p_cyl_butter_filtfilt(1:DSF:end); 
theta_DSF= theta_deg(1:DSF:end);

%% pressure FFT
len = length(p_DSF);
Y = fft(p_DSF/len);
N = 10;

% harmonic 0
c0 = abs(Y(1));

% preallocation
cn = zeros(N,1);
phi = zeros(N,1);

for n = 1:N
    cn(n) = 2*abs(Y(n+1));
    phi(n) = angle(Y(n+1)) + pi/2 ;
end
   
% Conversions 
theta_rad = deg2rad(theta_DSF);     
rpm = 2000;
omega = rpm * 2*pi/60;              % engine speed
t = theta_rad / omega; 
p_fourier = c0 * ones(size(theta_rad));
m = 2 ;  % 4 strokes
for n = 1:N
    p_fourier = p_fourier + cn(n)*sin(n/m* omega .*t+ phi(n));
end
%% plots
figure
plot(theta_DSF, p_DSF, 'b', 'LineWidth',1.5); hold on
plot(theta_DSF, p_fourier, 'r--', 'LineWidth',1.5);
xlabel('theta [°CA]')
ylabel('pressure [bar]')
title('Fourier approximation of the experimental pressure k=10')
legend('experimental','Fourier approx')
xlim([0 720])
grid on

%% Tangential pressure and Torque computation
V_d = V_d * 10^-6 /4;
A_piston = A_piston *10^(-6);  % in m^2
r = r * 10^-3; % in m
m_pis = 1.46 ;  %Piston mass 
m_rod = 1.40  ; %Conrod mass 
m_crk = 0.85 ;  %Crankpin mass 
m_rec = m_pis + 0.38 *m_rod;  %Recipr. masses
m_rot = m_crk + 0.62 *m_rod;     %Rotating masses
pc=1.01325*ones(size(theta_DSF))'; %Crankcase pressure
cos_beta = sqrt(1-lambda^(2)*(sin(theta_rad)).^2); % Calculate the angle beta
F_i = -m_rec * omega ^2 * r * (cos(theta_rad)+ lambda * cos(2.* theta_rad) / cos_beta);
bet_rad = asin(lambda * sin(theta_rad));  % Calculate the angle beta in radians
p_i = F_i / (A_piston)/10^5;  % bar
p_eff = p_DSF + p_i'-pc;   % bar     
P = p_eff * A_piston *10^5 ; % N
F = P' ./ cos (bet_rad);               % force along conrod
Ft = F .* sin(bet_rad + theta_rad);     % force tangent to the shaft
Ms = Ft * r ;              % single cylinder torque
Pt = Ms ./ ((V_d)/2)  /10^5 ;  % tangential pressure [bar]

% Effective pressure plot
figure(20)
plot (theta_DSF, p_eff ,'LineWidth',1.5)
xlabel('Crank angle [°CA]', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Effective Pressure [bar]', 'FontSize', 14, 'FontWeight', 'bold')
title('Effective Pressure over Crank Angle', 'FontSize', 20)
set(gca, 'FontSize', 12)
xlim ([0 720])
grid on; grid minor; box on

% Tangential pressure plot
figure
plot (theta_DSF, Pt,  'LineWidth',1.5)
xlabel('Crank angle [°CA]', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Tangential Pressure [bar]', 'FontSize', 14, 'FontWeight', 'bold')
title('Tangential Pressure over Crank Angle', 'FontSize', 20)
set(gca, 'FontSize', 12)
xlim ([0 720])
grid on; grid minor; box on

%% FFT torque
len_T = length(Ms);
Y = fft(Ms/len_T);
N = 10;

% harmonic 0
c0_T = abs(Y(1));

% preallocation
ncols = 2;
nrows = ceil(N/ncols);
cn_T = zeros(N,1);
phi = zeros(N,1);
T_fourier = c0_T * ones(size(theta_rad));
c0_T = c0_T * ones(size(theta_DSF));

% Plots
figure
subplot (nrows,ncols,1)
plot (theta_DSF, c0_T)
title('Harmonic order 0', 'FontSize', 10)
xlim([0 720])
grid on
for n = 1:N
    cn_T(n) = 2*abs(Y(n+1));
    phi_T(n) = angle(Y(n+1)) + pi/2 ;
    subplot (nrows+1,ncols,n+1)
    hold on
    harm = cn_T(n)*sin(n/m* omega .*t+ phi_T(n));
    T_fourier = T_fourier +  harm;
    plot (theta_DSF, harm)
    title(sprintf('Harmonic order %g', (n)/2), 'FontSize', 10)
    xlim([0 720])
    grid on
end
% avg torque
T_avgFourier = mean(T_fourier)*ones(size(theta_DSF));

%% torque plot 
figure
plot(theta_DSF, Ms, 'b', 'LineWidth',1.5); hold on
plot(theta_DSF, T_fourier, 'r--', 'LineWidth',1.5);
plot(theta_DSF, T_avgFourier, 'g--', 'LineWidth',1.5);
xlabel('Crank angle [°CA]', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Torque [Nm]', 'FontSize', 14, 'FontWeight', 'bold')
title('Torque for the single-cylinder k=10', 'FontSize', 20)
set(gca, 'FontSize', 12)
legend('Instantaneous torque', 'Fourier approximation', 'Average torque')
xlim ([0 720])
grid on; grid minor; box on

%% Fourier spectrum
figure
cn_T = cn_T(:).';  
cn_T = [c0_T(1), cn_T(1:end)];
harmonic_order = (0 : N)/2;
stem ( harmonic_order, cn_T)
xlabel('Harmonic order k', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('|Y|', 'FontSize', 14, 'FontWeight', 'bold')
title('Amplitude spectrum (torque) for the single cylinder', 'FontSize', 20)
grid on

%% Torque of others cylinders
% 3th
Ms_3 = circshift(Ms, 180);
figure
plot(theta_DSF, Ms_3, 'b', 'LineWidth',1.5); hold on
xlabel('Crank angle [°CA]', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Torque [Nm]', 'FontSize', 14, 'FontWeight', 'bold')
title('Torque for the cylinder 3', 'FontSize', 20)
set(gca, 'FontSize', 12)
xlim ([0 720])
grid on; grid minor; box on
% 4th
Ms_4 = circshift(Ms_3, 180);
figure
plot(theta_DSF, Ms_4, 'b', 'LineWidth',1.5); hold on
xlabel('Crank angle [°CA]', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Torque [Nm]', 'FontSize', 14, 'FontWeight', 'bold')
title('Torque for the cylinder 4', 'FontSize', 20)
set(gca, 'FontSize', 12)
xlim ([0 720])
grid on; grid minor; box on
% 2 nd
Ms_2 = circshift(Ms_4, 180);
figure
plot(theta_DSF, Ms_2, 'b', 'LineWidth',1.5); hold on
xlabel('Crank angle [°CA]', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Torque [Nm]', 'FontSize', 14, 'FontWeight', 'bold')
title('Torque for the cylinder', 'FontSize', 20)
set(gca, 'FontSize', 12)
xlim ([0 720])
grid on; grid minor; box on


%% Multi cylinder torque analysis
T_4cyl = Ms + Ms_2 + Ms_3 + Ms_4; % Total torque of multi cylinder engine
T_avg = mean(T_4cyl);
figure
plot(theta_DSF, T_4cyl, 'b', 'LineWidth',1.5); hold on
plot(theta_DSF, T_avg *ones(size(theta_DSF)),'r', 'LineWidth',1.5); hold on
xlabel('Crank angle [°CA]', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Torque [Nm]', 'FontSize', 14, 'FontWeight', 'bold')
title('Torque for the multi-cylinder engine', 'FontSize', 20)
set(gca, 'FontSize', 12)
xlim([0 720])
legend('Instantaneous torque', 'Average torque' )
grid on; grid minor; box on


% FFT torque multi cylinder
T_4 = zeros(720,4);
T_4(:,1) = Ms;
T_4(:,2) = Ms_2;
T_4(:,3) = Ms_3;
T_4(:,4) = Ms_4;

Y_T4 = zeros(720,4);
for j = 1:4
    Y_T4(:,j) = fft(T_4(:,j))/720;
end

N = 10;

% 0 Components
c0_T4 = abs(Y_T4(1,:));

% Output Fourier
T_fourier4 = zeros(720,4);
for j = 1:4
    T_fourier4(:,j) = c0_T4(j) * ones(720,1);
end
ncols = 2;
nrows = ceil(N/ncols);
fig_phasor = figure('Units','pixels','Position',[100 100 1200 800]);
hold on
axis equal
grid on
title('Phasor diagrams')

fig_harm = figure('Units','pixels','Position',[100 100 1200 800]);
hold on
grid on
title('Harmonic contributions','FontSize', 12)
xlim([0 720])

cn_T4  = zeros(N,4);
phi_T4 = zeros(N,4);
harm_T4 = zeros(720,4);
for n = 1:N
    figure(fig_phasor)
    subplot(nrows, ncols, n)
    hold on
    title(sprintf('order %g',n/2))
    axis equal
    for j = 1:4

        cn_T4(n,j)  = 2 * abs(Y_T4(n+1,j));
        phi_T4(n,j) = angle(Y_T4(n+1,j)) + pi/2;
        harm_T4(:,j) = cn_T4(n,j) * sin(n/m * omega .* t + phi_T4(n,j));
        T_fourier4(:,j) = T_fourier4(:,j) + harm_T4(:,j);              

        x  = [0, cn_T4(n,j)*cos(phi_T4(n,j))];
        y  = [0, cn_T4(n,j)*sin(phi_T4(n,j))];

        ang = 0:0.0001:2*pi;
        xp  = cn_T4(n,j) * cos(ang);
        yp  = cn_T4(n,j) * sin(ang);

        plot(xp,yp); hold on;
        plot(x,y); grid on;

    end
    figure(fig_harm)
    subplot(nrows,ncols,n)
    hold on
    title(sprintf('Harmonic order %g', n/2), 'FontSize', 12)
    xlabel('Crank angle °[CA]')
    ylabel('Torque [Nm')
    grid on
    for k = 1:4
        plot (theta_DSF, harm_T4 (:,k))
        xlim([0 720])
    end
end

%% Multi cylinder
T_multi = Ms + Ms_2 + Ms_3 + Ms_4;
% FFT torque
len_T = length(T_multi);
Y_multi = fft(T_multi/len_T);
N = 10;

% Calculate the average torque for the multi-cylinder engine
T_avg_multi = mean(T_multi)*ones(size(theta_DSF));

% harmonic 0
c0_Tmulti = abs(Y_multi(1));

% preallocation
cn_T_multi = zeros(N,1);
phi_multi = zeros(N,1);
T_fourier_multi = c0_Tmulti * ones(size(theta_rad));
for n = 1:N
    cn_T_multi(n) = 2*abs(Y_multi(n+1));
    phi_multi(n) = angle(Y_multi(n+1)) + pi/2 ;
    harm_multi = cn_T_multi(n)*sin(n/m* omega .*t+ phi_T(n));
    T_fourier_multi = T_fourier_multi +  harm_multi;
end
%% torque
figure
plot(theta_DSF, T_multi, 'b', 'LineWidth',1.5); hold on
plot(theta_DSF, T_fourier_multi, 'r--', 'LineWidth',1.5); hold on
plot(theta_DSF, T_avg_multi, 'm', 'LineWidth',1.5);
xlabel('Crank angle', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('Torque', 'FontSize', 14, 'FontWeight', 'bold')
title('Torque for the multi cylinder N=10', 'FontSize', 20)
set(gca, 'FontSize', 12)
legend('Instantaneous torque', 'Fourier approximation', 'Average torque')
xlim([0 720])
grid on; grid minor; box on

%% Stem
figure      
cn_T_multi = cn_T_multi(:).';   
cn_T_multi = [c0_Tmulti, cn_T_multi(1:end)];
stem ( harmonic_order, cn_T_multi)
xlabel('Harmonic order k', 'FontSize', 14, 'FontWeight', 'bold')
ylabel('|Y|', 'FontSize', 14, 'FontWeight', 'bold')
title('Amplitude spectrum (torque) for the multi cylinder', 'FontSize', 20)
grid on

%% REPORT 3 FLYWHEEL DIMENSIONING

p_DSF=circshift(p_DSF,360);
% Shift Pressure
Pt=circshift(Pt,360);
% Shift Torque and Theta
Ms_shifted = circshift(Ms, 360); % Shift left to start at expansion

figure
plot(theta_DSF,p_DSF)
title('gas pressure')
xlabel('theta [°CA]')
ylabel('gas pressure [bar]')
grid on, zoom on
figure
plot(theta_DSF,Pt)
title('tangential pressure')
xlabel('theta [°CA]')
ylabel('tangential pressure [bar]')
grid on, zoom on
rho_met= 7700; %value of density of the flywheel
m_alt= m_pis+m_rod*0.38; %reciprocating mass [kg]
m_rot= m_crk+m_rod*0.62; % equivalent rotating masses []kg/dm^3]
w_max= 3850*2*pi/60; %max angular speed
w_min= 1400*2*pi/60; %min angular speed
w_avg= 2000*2*pi/60; % average angular speed
delta= ((w_max-w_min)/w_avg)/100; %Kinematic irregularity
if delta>0.01
    delta=0.01;
end
V_xDSF=(V_x(1:DSF:end))';
V_xDSF = V_xDSF(1:720);
% 2. Calculate Resistant Torque (Mr)
% For steady state, Mr is constant and equals the average Shaft Torque
Mr = mean(Ms); 
Mr = Mr * ones(size(Ms));
delta_W=ones(size(theta_rad));

% calculate works
Ws = cumtrapz(theta_rad,Ms_shifted);
Wr = cumtrapz(theta_rad, Mr); % Calculate work done for each angle
delta_W=Ws-Wr;

% find min and max work variation
delta_Wmax=max(delta_W);
delta_Wmin=min(delta_W);
csi=(delta_Wmax+abs(delta_Wmin))/(imep_n*V_d);

% Required flywheel inertia and diameter
J_eng= m_rot*(r)^2;
J_tot=csi*imep_n*V_d/(delta*w_avg^2);
J_flyw=J_tot-J_eng;
D_flyw=(320*J_flyw/(pi*rho_met))^0.2;
w_flyw= (1/10)*D_flyw;
fprintf('Required Flywheel Inertia: %.4f kg m^2\n', J_flyw);
fprintf('Calculated Flywheel Diameter: %.4f m\n', D_flyw);

% Table
tr=1.708856*ones(size(theta_DSF));
V_x=V_x(1:end-1);
p_filtered=p_filtered(1:DSF:end);
theta_DSF = theta_DSF';
bet_deg = (rad2deg(bet_rad))';
p_i = p_i';
Ms = Ms';
Mr = Mr';
Pt = Pt';
tr = tr';
Ws = Ws';
Wr = Wr';
delta_W = delta_W';
p_resistance = (Mr / (V_d) / 10^5);
% Convert to bar for plotting (divide by 10^5)
Ws_norm = (Ws ./ V_d) / 10^5; 
Wr_norm = (Wr ./ V_d) / 10^5;
DeltaW_norm = (delta_W ./ V_d) / 10^5;
Q = table( theta_DSF, bet_deg, V_xDSF, p_filtered, pc, p_i, p_eff, Ms,Pt,Mr, tr,   Ws, Wr, p_resistance);

%% Flywheel Dimensioning: Work and Pressure plot
figure 
plot (theta_DSF,Pt, 'b', 'LineWidth',1.5) 
hold on
plot (theta_DSF,Ws_norm, 'g', 'LineWidth',1.5) 
hold on
plot (theta_DSF,Wr_norm, 'r', 'LineWidth',1.5) 
hold on
plot (theta_DSF,p_resistance, 'y', 'LineWidth',1.5)
hold on
legend('Tangential Press. (Pt)', 'Shaft Work (Ws)', 'Resistant Work (Wr)', 'Resistant Press. (Pr)');
xlabel('Crank Angle [deg]');
ylabel('Normalized Magnitude [bar]');
title('Flywheel Dimensioning: Work and Pressure');
xlim([0 720])
grid on
% (Starting Speed) in rad/s
%  w_avg as initial speed
w_start_rpm = 2000;
w_start_rad = w_start_rpm * (2*pi/60); 

% 2. speed (rad/s)
% NO FLYWHEEL
omega_nofly_rad = sqrt(w_start_rad^2 + (2/J_eng) .* delta_W);

% WITH FLYWHEEL
omega_fly_rad   = sqrt(w_start_rad^2 + (2/J_tot) .* delta_W);

% 3. in RPM
omega_nofly_rpm = omega_nofly_rad * (60/(2*pi));
omega_fly_rpm   = omega_fly_rad   * (60/(2*pi));

% mean values
mean_nofly = mean(omega_nofly_rpm);
mean_fly   = mean(omega_fly_rpm);

%% PLOT with and without flywheel
figure
% PLOT 1: NO FLYWHEEL 
plot(theta_DSF, omega_nofly_rpm, 'b', 'LineWidth', 2); hold on;
% Starting speed 
yline(w_start_rpm, 'r--', 'LineWidth', 2); 
% Average speed 
yline(mean_nofly, 'g:', 'LineWidth', 2); 

title('Angular speed without flywheel', 'FontSize', 14, 'Color', 'b');
subtitle('speed');
ylabel('Angular speed [rpm]', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('Theta [°CA]', 'FontSize', 10, 'FontWeight', 'bold');
legend('Instantaneous shaft speed', 'Starting speed', 'Average speed', ...
       'Location', 'best');
grid on; grid minor; box on;
xlim([0 720]);
ylim([1000 10000]); 

% --- PLOT 2: WITH FLYWHEEL ---
figure 

plot(theta_DSF, omega_fly_rpm, 'b', 'LineWidth', 2); hold on;
% Starting speed
yline(w_start_rpm, 'g:', 'LineWidth', 2);
% Average speed
yline(mean_fly, 'r--', 'LineWidth', 2); 

title('Angular speed with flywheel', 'FontSize', 14, 'Color', 'b');
subtitle('speed');
ylabel('Angular speed [rpm]', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('Theta [°CA]', 'FontSize', 10, 'FontWeight', 'bold');
legend('Instantaneous shaft speed', 'Starting speed', 'Average speed', ...
       'Location', 'best');
grid on; grid minor; box on;
xlim([0 720]);
ylim([1990 2030]);


%  T_avg is the average torque


%  T_4cyl is the total torque

Ws_multi = cumtrapz(theta_rad, T_4cyl);
Wr_multi = cumtrapz(theta_rad, T_avg*ones(size(theta_rad)));
DeltaW_multi = Ws_multi - Wr_multi;
csi_multi = max(DeltaW_multi) - min(DeltaW_multi);
J_tot_multi = csi_multi / ((w_avg)^2 * delta);
D_fly_multi = ((J_tot_multi * 320) / (pi * rho_met))^(1/5);
fprintf('\n Required Flywheel Inertia multi-cylinder: %.4f kg m^2\n', J_tot_multi);
fprintf('Calculated Flywheel Diameter multi-cylinder: %.4f m\n', D_fly_multi);
pt_4cyl=T_4cyl/V_d/2 *10^(-5) *i;
pt_2=Ms_2/V_d/2 *10^(-5) *i;
pt_3=Ms_3/V_d/2 *10^(-5) *i;
pt_4=Ms_4/V_d/2 *10^(-5) *i;
figure
plot(theta_DSF,pt_4cyl,'k','LineWidth',3)
hold on
plot(theta_DSF,Pt,'LineWidth',1)
hold on
plot(theta_DSF,pt_2,'LineWidth',1)
hold on
plot(theta_DSF,pt_3,'LineWidth',1)
hold on
plot(theta_DSF,pt_4,'LineWidth',1)
hold on
grid on
zoom on
legend('Engine','1','2','3','4')

%% Plot Multi-cyl

%omega flywheel multi-cyl
omega_flyMulti_rad   = sqrt(w_start_rad^2 + (2/J_tot_multi) .* DeltaW_multi);
mean_flyMulti   = mean(omega_flyMulti_rad);

figure 
plot(theta_DSF, omega_flyMulti_rad, 'b', 'LineWidth', 2); hold on;
% Starting speed
yline(w_start_rad, 'g:', 'LineWidth', 2);
% Average speed
yline(mean_flyMulti, 'r--', 'LineWidth', 2); 

title('Multi-cyl flywheel', 'FontSize', 14, 'Color', 'b');
subtitle('speed');
ylabel('angular speed [rad]', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('\theta [°CA]', 'FontSize', 10, 'FontWeight', 'bold');
legend('Instantaneous shaft speed', 'starting speed', 'average speed', ...
       'Location', 'best');
grid on; grid minor; box on;
xlim([0 180]);

% Injection
inj = mean(ifile.INJ1.data, 2);

% delta Work
[max_deltay,index_max] = max(delta_W);
max_deltax =  theta_DSF (index_max);
figure
plot(theta_DSF, delta_W)
title('Delta work', 'FontSize', 14, 'Color', 'b');
subtitle('speed');
ylabel('Delta work', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('\theta [°CA]', 'FontSize', 10, 'FontWeight', 'bold');
grid on; grid minor; box on;
hold on
xlim ([0 720])
plot(max_deltax,max_deltay,'ro','MarkerSize',10,'LineWidth',2)

%% Heat Release - part 4

% Start of injection
SOI = find(inj>1,1,"first")/10;
delta_p = zeros(size(theta_DSF));
delta_pV = zeros(size(theta_DSF));
p_motored = p_cyl_butter_filtfilt;
m = 1.3;
idx = 3500;
V_ref= V_x(idx);
p_ref = p_cyl_butter_filtfilt(idx);

for h= 3500:7199
    delta_p(h) =p_cyl_butter_filtfilt(h+1) - p_cyl_butter_filtfilt(h);
    delta_pV(h) = p_cyl_butter_filtfilt(h)*((V_ref/V_x(h+1))^m -1);
    p_motored(h+1) = p_ref*((V_ref/V_x(h+1))^m);
end
mf_tot = 0;
delta_pc = delta_p - delta_pV;      % combustion pressure

figure(99)
yyaxis left
plot(theta_deg,p_cyl_butter_filtfilt,'b-'); hold on
plot(theta_deg,p_motored,'r--')
xlim ([320 430])
title('In-cylinder pressure and mass fraction burned')
ylabel('Pressure [bar]', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('Theta [°CA]', 'FontSize', 10, 'FontWeight', 'bold');
xlim([0 720])
yyaxis right


m_mix = T.m_air(7)+T.m_EGR(7);
R_mix = T.R_mix(7);
temperature = (p_cyl_butter_filtfilt .* V_x'*10^-1) ./ (m_mix * R_mix);   % [K]
gamma = 1.338 -6 * 10^(-5).*temperature + 10^(-8).*temperature.^2;    
% 4) derivative of p and V
dV_dtheta = gradient(V_x*10^-6, theta)';    % m^3 / deg
dp_dtheta = gradient(p_cyl_butter_filtfilt*10^5, theta);  % MPa / deg

% compute HRR 
HRR=ones(length(theta),1);
for j= 2:length(theta)
    HRR(j) = 1.05*gamma(j)/ (gamma(j)-1) * p_cyl_butter_filtfilt(j) *(V_x(j)-V_x(j-1))+1/(gamma(j)-1)* V_x(j)*(p_cyl_butter_filtfilt(j)-p_cyl_butter_filtfilt(j-1));
end

% Window indices
EOC = 450;
theta_deg= theta_deg';
% Indices inside the integration window
idx = (theta_deg >= SOI) & (theta_deg <= EOC);
Q_lhv = 42.5 * 10^6;      %J/kg
theta_w = theta_deg(idx);
HRR_w   = HRR(idx);
% Compute Xb only in the combustion window
Xb = zeros(size(theta_deg));
Xb(idx) = cumtrapz(theta_w, HRR_w) / (T.m_fuel(7)* Q_lhv);
figure
plot(theta_deg,HRR,'r')
title('HRR', 'FontSize', 14, 'FontWeight', 'bold')
xlabel('Crank angle [°CA]', 'FontSize', 10, 'FontWeight', 'bold')
ylabel('Net heat release rate', 'FontSize', 10, 'FontWeight', 'bold')
xlim ([0 720])
grid on
% EMFB10/50/90 and SOC/SOI 
MFB10 = theta_deg(find(Xb >= 0.10, 1));
MFB50 = theta_deg(find(Xb >= 0.50, 1));
MFB90 = theta_deg(find(Xb >= 0.90, 1));

% SOC: first point where XB>small threshold
SOC_idx = find(Xb > 0.01, 1);
SOC = theta_deg(SOC_idx);
ID = SOC - SOI;     % ignition delay

fprintf('MFB10=%.2f deg, MFB50=%.2f deg, MFB90=%.2f deg, SOC=%.2f deg\n,SOI=%.2f deg\n,ID=%.2f deg\n', MFB10, MFB50, MFB90, SOC, SOI, ID);


% MFB10 line
figure(99)
plot(theta_deg,Xb,'g--')
xlim ([320 430])
ylabel('Mass burned fraction', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('\theta [°CA]', 'FontSize', 10, 'FontWeight', 'bold');
set(gca, 'YColor', 'g');
hold on, grid on, box on
% Plot injection data
xline(MFB10, 'k--', 'LineWidth', 1.2, ...
       'Label', 'MFB10', 'LabelVerticalAlignment', 'bottom', ...
       'LabelHorizontalAlignment', 'center');

% MFB50 line
xline(MFB50, 'm--', 'LineWidth', 1.2, ...
       'Label', 'MFB50', 'LabelVerticalAlignment', 'bottom', ...
       'LabelHorizontalAlignment', 'center');

% MFB90 line
xline(MFB90, 'c--', 'LineWidth', 1.2, ...
       'Label', 'MFB90', 'LabelVerticalAlignment', 'bottom', ...
       'LabelHorizontalAlignment', 'center');
% SOI line
xline(SOI, 'c--', 'LineWidth', 1.2, ...
       'Label', 'SOI', 'LabelVerticalAlignment', 'bottom', ...
       'LabelHorizontalAlignment', 'center');
% SOC line
xline(SOC, 'c--', 'LineWidth', 1.2, ...
       'Label', 'SOC', 'LabelVerticalAlignment', 'bottom', ...
       'LabelHorizontalAlignment', 'center');

% Injection current
figure
plot(theta_deg, inj, 'r:', 'LineWidth', 1.5);
xlabel('Crank angle', 'FontSize', 10, 'FontWeight', 'bold')
ylabel('Injection current', 'FontSize', 10, 'FontWeight', 'bold')
title('Injection current', 'FontSize', 14, 'FontWeight', 'bold')
xlim([320 430])
grid on

