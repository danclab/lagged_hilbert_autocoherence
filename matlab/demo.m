

%% Simulated signal
% Generate a simulated signal with 3 bursts at 15Hz with 5 cycles
% and an oscillation at 20Hz, plus white noise

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
signal=s1+s2+(rand(1,length(time))*2-1);

figure;
subplot(3,1,1);
plot(time,s1);
xlim([time(1) time(end)]);
subplot(3,1,2);
plot(time,s2);
xlim([time(1) time(end)]);
subplot(3,1,3);
plot(time,signal);
xlim([time(1) time(end)]);
xlabel('Time (s)');


%% Lagged Hilbert coherence
% Lagged Hilbert coherence starts with bandpass filtering using
% multiplication by a Gaussian kernel in the frequency domain

% Frequencies to evaluate (just to compute desired frequency
% resolution in this step)
freqs=linspace(5,100,100);

% Determine the frequency resolution
df = diff(freqs(1:2));

% Zero-pad the signal - or is mean-pad better?
pad=zeros(size(time));
padd_signal = [pad, signal, pad];

% Fourier transform the padded signal
signal_fft = rfft(padd_signal,length(padd_signal), 2);
fft_frex = linspace(0, srate/2, length(signal_fft));

% Kernel width for multiplication
sigma = df * .5;

% Center pass-band frequency
freq=15;

% Gaussian kernel centered on frequency with width defined
% by requested frequency resolution
kernel = exp(-((fft_frex - freq) .^ 2 / (2.0 * sigma ^ 2)));

% Multiply Fourier-transformed signal by kernel
fsignal_fft = signal_fft .* kernel;
% Reverse Fourier to get bandpass filtered signal
f_signal = irfft(fsignal_fft, length(padd_signal), 2);

figure();
subplot(5,1,1)
plot(time,signal)
xlim([time(1), time(end)])
xlabel('Time (s)')
ylabel('Amplitude')

subplot(5,1,2)
plot(fft_frex,signal_fft(1:length(fft_frex)))
xlim([0,50])
xlabel('Frequency (Hz)')
ylabel('Amplitude')

subplot(5,1,3)
plot(fft_frex,kernel,'k')
xlim([0,50])
xlabel('Frequency (Hz)')
ylabel('Amplitude')

subplot(5,1,4)
plot(fft_frex,fsignal_fft(1:length(fft_frex)))
xlim([0,50])
xlabel('Frequency (Hz)')
ylabel('Amplitude')

subplot(5,1,5)
plot(time,f_signal(length(pad)+1:end-length(pad)))
xlim([time(1), time(end)])
xlabel('Time (s)')
ylabel('Amplitude')



% Get analytic signal of bandpass filtered data (phase and amplitude)
analytic_signal = hilbert(f_signal);
% Cut off padding
analytic_signal=analytic_signal(:, length(time)+1:2 * length(time));

figure();
subplot(4,1,1);
plot(time,signal);
xlim([time(1) time(end)]);
xlabel('Time (s)');
ylabel('Amplitude');

subplot(4,1,2);
plot(time,f_signal(:, length(time)+1:2 * length(time)));
xlim([time(1) time(end)]);
xlabel('Time (s)');
ylabel('Amplitude');

subplot(4,1,3);
plot(time,abs(analytic_signal));
xlim([time(1) time(end)]);
xlabel('Time (s)');
ylabel('Amplitude');

subplot(4,1,4);
plot(time,angle(analytic_signal));
xlim([time(1) time(end)]);
xlabel('Time (s)');
ylabel('Phase');




% Evaluate lagged coherence at a lag of 3 cycles
lag=3;

% Duration of this lag in s
lag_dur_s = lag / freq;
% Number of evaluations
n_evals = floor(T / lag_dur_s);
% Remaining time
t_diff = T - (n_evals * lag_dur_s);

% Start time
start_time = time(1);
% Evaluation times (ms)
eval_times = linspace(start_time, T - t_diff, n_evals + 1); 
eval_times = eval_times(1:end-1);
% Evaluation time points
eval_pts = knnsearch(time', eval_times');

figure()
subplot(4,1,1)
plot(time,signal)
hold on
plot(eval_times,signal(eval_pts),'ro')
xlabel('Time (s)')
ylabel('Amplitude')

subplot(4,1,2)
plot(time,f_signal(length(time)+1:2*length(time)))
hold on
plot(eval_times,f_signal(length(time)+eval_pts),'ro')
xlabel('Time (s)')
ylabel('Amplitude')

subplot(4,1,3)
plot(time,abs(analytic_signal))
hold on
plot(eval_times,abs(analytic_signal(eval_pts)),'ro')
xlabel('Time (s)')
ylabel('Amplitude')

subplot(4,1,4)
plot(time,angle(analytic_signal))
hold on
plot(eval_times,angle(analytic_signal(eval_pts)),'ro')
xlabel('Time (s)')
ylabel('Phase')

% Analytic signal at n=0...n_evals-1 evaluation points
f1 = analytic_signal(eval_pts(1:end-1));
% Analytic signal at n=1...n_evals evaluation points
f2 = analytic_signal(eval_pts(2:end));
% calculate the phase difference and amplitude product
phase_diff = angle(f2) - angle(f1);
amp_prod = abs(f1) .* abs(f2);

figure();
subplot(2,1,1)
plot(eval_times(1:end-1),phase_diff)
xlim([time(1), time(end)])
xlabel('Time (s)')
ylabel('Phase difference')

subplot(2,1,2)
plot(eval_times(1:end-1),amp_prod)
xlim([time(1), time(end)])
xlabel('Time (s)')
ylabel('Amplitude product')

% Numerator - sum is over evaluation points
num = abs(sum(amp_prod .* exp(1i * phase_diff)));
disp(num)

% Scaling factor - sum is over evaluation points
f1_pow = f1.^2;
f2_pow = f2.^2;
denom = sqrt(sum(abs(f1_pow)) * sum(abs(f2_pow)));
disp(denom);

lc = num / denom;
disp(lc);



% This was evaluated starting at time t=0 and looking 3 cycles ahead, 
% but what about the points in between?

% Number of points between the first and next evaluation time points
n_range = eval_pts(2) - eval_pts(1);

% Analytic signal at n=0...n_evals-1 evaluation points, and m=0..n_range time points in between
f1 = analytic_signal(eval_pts(1:end-1) + (0:n_range-1));

% Analytic signal at n=1...n_evals evaluation points, and m=0..n_range time points in between
f2 = analytic_signal(eval_pts(2:end) + (0:n_range-1));

% Calculate the phase difference and amplitude product
phase_diff = angle(f2) - angle(f1);
amp_prod = abs(f1) .* abs(f2);

% Lagged coherence
num = abs(sum(amp_prod .* exp(1i * phase_diff),1));
f1_pow = f1.^2;
f2_pow = f2.^2;
denom = sqrt(sum(abs(f1_pow),1) .* sum(abs(f2_pow),1));
lc = num / denom;

% Average over the time points in between evaluation points
lc = mean(lc);
disp(lc);


% Evaluate at 2-10 lag cycles
lags = 2:.5:10;
% Evaluate at 15 and 20 Hz
freqs = [15, 20];

figure();
for f_idx = 1:length(freqs)
    freq = freqs(f_idx);
    lcs = zeros(size(lags));
    for l_idx = 1:length(lags)
        [num, denom, lcs(l_idx)] = lagged_hilbert_coh_demo(signal, srate, freq, lags(l_idx), df);
    end

    plot(lags, lcs, 'DisplayName', sprintf('%dHz', freq))
    hold on
end
hold off
legend
xlabel('Lag (cycles)')
ylabel('Lagged coherence')




% What about at a frequency where we know there is only noise?
[num, denom, lc] = lagged_hilbert_coh_demo(signal, srate, 50, 3, df);
disp(lc);




% What's going on?
[num1, denom1, lc1] = lagged_hilbert_coh_demo(signal, srate, 15, 3, df);
[num2, denom2, lc2] = lagged_hilbert_coh_demo(signal, srate, 50, 3, df);

fprintf('15Hz: numerator=%f, denominator=%f, lc=%f\n', mean(num1), mean(denom1), lc1);
fprintf('50Hz: numerator=%f, denominator=%f, lc=%f\n', mean(num2), mean(denom2), lc2);




% At 40Hz, the numerator and denominator are both low, but nearly exactly the same
% This is due to amplitude correlations introduced by bandpass filtering in a freq
% range with low power (https://journals.physiology.org/doi/full/10.1152/jn.00851.2013)
% Here's the effect over a range of frequencies

% Evaluate at 2-10 lag cycles
lags = 2:.5:10;
% Evaluate at 5-100 Hz
freqs = linspace(5, 100, 100);

lcs = zeros(length(freqs), length(lags));
for f_idx = 1:length(freqs)
    freq = freqs(f_idx);
    for l_idx = 1:length(lags)
        [num, denom, lcs(f_idx, l_idx)] = lagged_hilbert_coh_demo(signal, srate, freq, lags(l_idx), df);
    end
end

figure();
contourf(lags, freqs, lcs, 100,'LineColor','none');
colorbar;
xlabel('Lag (cycles)');
ylabel('Frequency (Hz)');





% The solution is to use the amplitude covariance of a surrogate 
% dataset as a threshold. We use an AR model to account for
% aperiodic temporal structure

% Compute threshold as 95th percentile of shuffled amplitude products
n_shuffles=1000;
amp_prods=ar_surr(signal, n_shuffles);
threshold = prctile(amp_prods, 95);

% Evaluate at 2-10 lag cycles
lags = 2:.5:10;
% Evaluate at 5-100 Hz
freqs = linspace(5, 100, 100);

lcs = zeros(length(freqs), length(lags));
for f_idx = 1:length(freqs)
    for l_idx = 1:length(lags)
        [num, denom, lc] = lagged_hilbert_coh_demo(signal, srate, freqs(f_idx), lags(l_idx), df);
        % Only consider lc if denominator greater than threshold
        if mean(denom) >= threshold
            lcs(f_idx, l_idx) = lc;
        end
    end
end

figure();
contourf(lags, freqs, lcs, 100,'LineColor','none');
colorbar;
xlabel('Lag (cycles)');
ylabel('Frequency (Hz)');







%% Put it all together
function [num, denom, lc]=lagged_hilbert_coh_demo(signal, srate, freq, lag, df)

    n_pts=length(signal);
    dt=1/srate;
    T = n_pts * dt;
    time=0:dt:T-dt;
    
    % Zero-pad the signal - or is mean-pad better?
    pad=zeros(size(time));
    padd_signal = [pad, signal, pad];

    % Fourier transform the padded signal
    signal_fft = rfft(padd_signal,length(padd_signal), 2);
    fft_frex = linspace(0, srate/2, length(signal_fft));

    % Kernel width for multiplication
    sigma = df * .5;

    % Gaussian kernel centered on frequency with width defined
    % by requested frequency resolution
    kernel = exp(-((fft_frex - freq) .^ 2 / (2.0 * sigma ^ 2)));

    % Multiply Fourier-transformed signal by kernel
    fsignal_fft = signal_fft .* kernel;
    % Reverse Fourier to get bandpass filtered signal
    f_signal = irfft(fsignal_fft, length(padd_signal), 2);

    analytic_signal = hilbert(f_signal);
    % Cut off padding
    analytic_signal=analytic_signal(:, length(time)+1:2 * length(time));

    % Duration of this lag in s
    lag_dur_s = lag / freq;
    % Number of evaluations
    n_evals = floor(T / lag_dur_s);
    % Remaining time
    t_diff = T - (n_evals * lag_dur_s);

    % Start time
    start_time = time(1);
    % Evaluation times (ms)
    eval_times = linspace(start_time, T - t_diff, n_evals + 1); 
    eval_times = eval_times(1:end-1);
    % Evaluation time points
    eval_pts = knnsearch(time', eval_times');
    
    % Number of points between the first and next evaluation time points
    n_range = eval_pts(2) - eval_pts(1);

    % Analytic signal at n=0...n_evals-1 evaluation points, and m=0..n_range time points in between
    f1 = analytic_signal(eval_pts(1:end-1) + (0:n_range-1));

    % Analytic signal at n=1...n_evals evaluation points, and m=0..n_range time points in between
    f2 = analytic_signal(eval_pts(2:end) + (0:n_range-1));

    % Calculate the phase difference and amplitude product
    phase_diff = angle(f2) - angle(f1);
    amp_prod = abs(f1) .* abs(f2);

    % Lagged coherence
    num = abs(sum(amp_prod .* exp(1i * phase_diff),1));
    f1_pow = f1.^2;
    f2_pow = f2.^2;
    denom = sqrt(sum(abs(f1_pow),1) .* sum(abs(f2_pow),1));
    lc = num / denom;

    % Average over the time points in between evaluation points
    lc = mean(lc);


end