

%% Generate a simulated signal with 3, bursts at 15Hz with 5 cycles
% and an oscillation at 20Hz, plus white noise

n_trials=100;

% Time step (s)
dt=0.001;
% Duration (s)
T=5;
% Time vector
time=0:dt:T-dt;
% Sampling rate
srate=1/dt;

% Burst frequency
f1 = 15;
% Length of bursts in cycles
f1_num_cycles=5;
% Number of bursts
f1_num_bursts=3;

% Oscillation frequency
f2 = 20;

signal=zeros(n_trials,length(time));

for t=1:n_trials

    % Burst signal
    s1=zeros(1,length(time));

    % Keep track of burst start and stop times so they
    % don't overlap
    burst_starts=[];
    burst_stops=[];
    while length(burst_starts)<f1_num_bursts
        % Burst duration in seconds
        dur_s=f1_num_cycles/f1;
        % Burst duration in time steps
        dur_pts=floor(dur_s/dt);

        % Random start and stop time
        start=randi(length(time)-dur_pts);
        stop=start+dur_pts;

        % Check that doesn't overlap with other bursts
        overlap=false;
        for k=1:length(burst_starts)
            other_start=burst_starts(k);
            other_stop=burst_stops(k);
            if (start >= other_start && start < other_stop) || (stop > other_start && stop <= other_stop)
                overlap=true;
                break;
            end
        end

        % Generate burst
        if ~overlap
            s1(start:stop)=sin(2. * pi * f1 * (time(start:stop)+randn()));
            burst_starts=[burst_starts start];
            burst_stops=[burst_stops stop];
        end
    end

    % Oscillatory signal
    s2=sin(2. * pi * f2 * (time+randn()));

    % Generated signal
    signal(t,:)=s1+s2+(rand(1,length(time))*2-1);
end

figure;
subplot(3,1,1);
plot(time,s1);
xlim([time(1) time(end)]);
subplot(3,1,2);
plot(time,s2);
xlim([time(1) time(end)]);
subplot(3,1,3);
plot(time,signal(end,:));
xlim([time(1) time(end)]);
xlabel('Time (s)');




% Evaluate at 2-10 lag cycles
lags = 1:0.5:10;

% Evaluate at 5-100 Hz
freqs = 5:1:100;

trial_lcs_hilbert=lagged_hilbert_autocoherence(signal, freqs, lags, srate);

figure();
contourf(lags, freqs, squeeze(mean(trial_lcs_hilbert,1)), 100,'LineColor','none');
set(gca,'clim',[0 1]);
colorbar;
xlabel('Lag (cycles)');
ylabel('Frequency (Hz)');


% Generate a simulated signal with just white noise

n_trials=100;

% Time step (s)
dt=.001;
% Duration (s)
T=5;
% Time vector
time=linspace(0,T,T/dt);
% Sampling rate
srate=1/dt;

signal=zeros(n_trials,length(time));
for t=1:n_trials
    signal(t,:)=rand(1,length(time))*2-1;
end

figure;
subplot(1,1,1);
plot(time,signal(1,:));
xlim([time(1) time(end)]);
xlabel('Time (s)');


% Evaluate at 2-10 lag cycles
lags = 1:0.5:10;

% Evaluate at 5-100 Hz
freqs = 5:1:100;

trial_lcs_hilbert=lagged_hilbert_autocoherence(signal, freqs, lags, srate);

figure();
contourf(lags, freqs, squeeze(mean(trial_lcs_hilbert,1)), 100,'LineColor','none');
set(gca,'clim',[0 1]);
colorbar;
xlabel('Lag (cycles)');
ylabel('Frequency (Hz)');
