function amp_prods = generate_surrogate(signal, n_shuffles)
%GENERATE_SURROGATE  Generate per-trial surrogate amplitude products via phase randomization
%
% Parameters
% ----------
% signal : matrix
%     Input signal, shape (n_trials, n_pts)
% n_shuffles : int
%     Number of surrogate realizations per trial
%
% Returns
% -------
% amp_prods : array
%     Shape: (n_trials, n_shuffles, n_pts - 1)

[n_trials, n_pts] = size(signal);
amp_prods = zeros(n_trials, n_shuffles, n_pts - 1);

parfor i = 1:n_trials
    x = signal(i, :);

    % Local variable for storing surrogates
    trial_amp_prods = zeros(n_shuffles, n_pts - 1);

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

    amp_prods(i, :, :) = trial_amp_prods;
end

end
