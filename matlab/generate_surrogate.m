function amp_prods = generate_surrogate(signal, n_shuffles, method)
%GENERATE_SURROGATE  Generate per-trial surrogate amplitude products via ARMA or phase randomization
%
% Parameters
% ----------
% signal : matrix
%     Input signal, shape (n_trials, n_pts)
% n_shuffles : int
%     Number of surrogate realizations per trial
% method : str
%     'arma' or 'phase'
%
% Returns
% -------
% amp_prods : array
%     Shape: (n_trials, n_shuffles, n_pts - 1)

if nargin < 3
    method = 'phase';
end

[n_trials, n_pts] = size(signal);
amp_prods = zeros(n_trials, n_shuffles, n_pts - 1);

parfor i = 1:n_trials
    x = signal(i, :);

    % Local variable for storing surrogates
    trial_amp_prods = zeros(n_shuffles, n_pts - 1);

    switch lower(method)
        case 'arma'
            % Mean adjust
            dataMeanAdjusted = x - mean(x);
            p = 1; % AR order

            % Autocovariance
            autoCov = xcov(dataMeanAdjusted, p, 'biased');

            % Yule-Walker equations
            R = toeplitz(autoCov(p+1:p+p));
            rho = autoCov(p+2:p+p+1);

            % Solve for AR coefficients
            arCoeff = R \ rho;

            % Simulate surrogates one-by-one
            for j = 1:n_shuffles
                x_sim = zeros(1, n_pts);
                x_sim(1:p) = x(1:p);
                for t = p+1:n_pts
                    x_sim(t) = -arCoeff' * x_sim(t-p:t-1) + randn;
                end
                x_sim = x_sim + mean(x);

                % Hilbert amplitude product
                analytic_rand_signal = hilbert(x_sim);
                trial_amp_prods(j, :) = abs(analytic_rand_signal(1:end-1)) .* ...
                                        abs(analytic_rand_signal(2:end));
            end

        case 'phase'
            % FFT of original
            fft_x = fft(x);
            amp = abs(fft_x(1:floor(n_pts/2) + 1));

            % Random phases for all shuffles
            rand_phase = exp(1i * (2 * pi * rand(n_shuffles, numel(amp))));

            % Apply to amplitude spectrum
            surrogates = zeros(n_shuffles, n_pts);
            for j = 1:n_shuffles
                surrogates(j, :) = irfft(amp .* rand_phase(j, :), n_pts, 2);
            end

            % Hilbert amplitude products in vectorized form
            analytic_rand_signal = hilbert(surrogates')';
            trial_amp_prods = abs(analytic_rand_signal(:, 1:end-1)) .* ...
                              abs(analytic_rand_signal(:, 2:end));

        otherwise
            error('Unknown method: %s', method);
    end

    amp_prods(i, :, :) = trial_amp_prods;
end

end
