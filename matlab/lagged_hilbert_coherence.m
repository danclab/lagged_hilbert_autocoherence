function lcs=lagged_hilbert_coherence(signal, freqs, lags, ...
    srate, varargin)

    % Parse inputs
    defaults = struct('df', -1, 'n_shuffles', 1000, 'type', 'coh',...
        'thresh_prctile', 95);  %define default values
    params = struct(varargin{:});
    for f = fieldnames(defaults)',
        if ~isfield(params, f{1}),
            params.(f{1}) = defaults.(f{1});
        end
    end

    n_trials = size(signal,1);
    n_pts = size(signal,2);
    dt=1/srate;
    T = n_pts * dt;
    time=0:dt:T-dt;
    
    % Check that epochs are long enough for requested frequencies and lags
    min_freq = min(freqs);
    max_lag = max(lags);
    lag_dur_s = max([max_lag / min_freq, 1 / srate]);
    min_epoch_len = 2 * lag_dur_s;

    if T < min_epoch_len
        error('Epoch length must be at least %.2fs to evaluate LHC at %.2f Hz and %.2f cycles', ...
              min_epoch_len, min_freq, max_lag);
    end

    % Bandpass filtering using multiplication by a Gaussian kernel
    % in the frequency domain
    n_freqs=length(freqs);
    n_lags=length(lags);
    
    % Determine the frequency resolution
    if params.df==-1
        df = diff(freqs(1:2));
    else
        df = params.df;
    end
        
    filtered_signal = bandpass(signal, [freqs(1) freqs(end)], srate);
    
    amp_prods=ar_surr(filtered_signal, params.n_shuffles);
    amp_prods=mean(amp_prods,3);
    thresh = prctile(amp_prods, params.thresh_prctile, 2);

    padd_signal = [zeros(size(signal)), signal, zeros(size(signal))];

    % Fourier transform the padded signal
    signal_fft = rfft(padd_signal,size(padd_signal,2), 2);
    fft_frex = linspace(0, srate/2, size(signal_fft,2));

    % Kernel width for multiplication
    sigma = df * .5;

    lcs=zeros(n_trials,n_freqs,n_lags);
    type=params.type;
    
    parfor f_idx=1:n_freqs
        freq=freqs(f_idx);
        
        % Gaussian kernel centered on frequency with width defined
        % by requested frequency resolution
        kernel = exp(-((fft_frex - freq) .^ 2 / (2.0 * sigma ^ 2)));

        % Multiply Fourier-transformed signal by kernel
        fsignal_fft = signal_fft .* kernel;
        % Reverse Fourier to get bandpass filtered signal
        f_signal = irfft(fsignal_fft, size(padd_signal,2), 2);

        % Get analytic signal of bandpass filtered data (phase and amplitude)
        analytic_signal = hilbert(f_signal')';
        % Cut off padding
        analytic_signal=analytic_signal(:, n_pts+1:2 * n_pts);

        for l_idx=1:n_lags
            lag=lags(l_idx);
            
            % Duration of this lag in s
            lag_dur_s = max([lag / freq, 1/srate]);
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

            % This was evaluated starting at time t=0 and looking 2 cycles ahead, 
            % but what about the points in between?

            % Number of points between the first and next evaluation time points
            n_range = eval_pts(2) - eval_pts(1);

            % Analytic signal at n=0...n_evals-1 evaluation points, and m=0..n_range time points in between
            f1=reshape(analytic_signal(:, eval_pts(1:end-1) + (0:n_range-1)),[n_trials,length(eval_pts)-1,n_range]);

            % Analytic signal at n=1...n_evals evaluation points, and m=0..n_range time points in between
            f2=reshape(analytic_signal(:, eval_pts(2:end) + (0:n_range-1)),[n_trials,length(eval_pts)-1,n_range]);

            % Calculate the phase difference and amplitude product
            phase_diff = angle(f2) - angle(f1);
            amp_prod = abs(f1) .* abs(f2);

            lc=0;
            if strcmp(type,'coh')
                % Lagged coherence
                num = squeeze(sum(amp_prod .* exp(1i * phase_diff),2));
                f1_pow = f1.^2;
                f2_pow = f2.^2;
                denom = squeeze(sqrt(sum(abs(f1_pow),2) .* sum(abs(f2_pow),2)));
                lc = abs(num ./ denom);
                lc(denom < repmat(thresh, [1, size(lc, 2)])) = 0;
            elseif strcmp(type,'plv')
                expected_phase_diff = lag * 2 * pi;
                num = squeeze(sum(exp(1i * (expected_phase_diff-phase_diff)),2));
                denom = length(eval_pts) - 1;
                lc = num ./ denom;
                f1_pow = f1.^2;
                f2_pow = f2.^2;
                amp_denom = squeeze(sqrt(sum(abs(f1_pow),2) .* sum(abs(f2_pow),2)));                
                % Threshold based on amplitude denominator
                lc(amp_denom < repmat(thresh, [1, size(lc, 2)])) = 0;
            elseif strcmp(type,'amp_coh')
                num = squeeze(sum(amp_prod,2));
                f1_pow = f1.^2;
                f2_pow = f2.^2;
                denom = squeeze(sqrt(sum(abs(f1_pow),2) .* sum(abs(f2_pow),2)));
                lc = abs(num ./ denom);
                lc(denom < repmat(thresh, [1, size(lc, 2)])) = 0;
            end
            
            % Average over the time points in between evaluation points
            lcs(:,f_idx,l_idx) = mean(lc,2);
        end
    end
    