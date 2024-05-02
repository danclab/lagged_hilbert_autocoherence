function amp_prods=ar_surr(signal, n_shuffles)

    n_trials = size(signal, 1);
    n_pts = size(signal, 2);

    amp_prods = zeros(n_trials, n_shuffles, n_pts-1);

    parfor i = 1:n_trials
        x = signal(i, :);

        % Estimate an AR model
        %mdl = arima(1, 0, 0);
        %mdl = estimate(mdl, x', 'Display', 'off');
        p = 1; % Order of AR model

        % Compute mean-adjusted data
        dataMeanAdjusted = x - mean(x);

        % Compute autocovariance
        autoCov = xcov(dataMeanAdjusted, p, 'biased');

        % Formulate the Yule-Walker equations
        R = toeplitz(autoCov(p+1:p+p));
        rho = autoCov(p+2:p+p+1);

        % Solve the Yule-Walker equations to get AR coefficients
        arCoeff = R\rho;

        %x_sim = simulate(mdl, n_pts, 'NumPaths', n_shuffles)';
        x_sim = zeros(n_shuffles, n_pts);
        for j = 1:n_shuffles
            % Initial values (you can replace these with specific initial values if desired)
            x_sim(j,1:p) = x(1:p);

            % Generate AR simulated data
            for t = p+1:n_pts
                x_sim(j,t) = -arCoeff' * x_sim(j,t-p:t-1) + randn;
            end

            % If the mean was adjusted, add it back to the simulated data
            x_sim(j,:) = x_sim(j,:) + mean(x);
        end
        
        padd_rand_signal = [zeros(size(x_sim)), x_sim, zeros(size(x_sim))];
        analytic_rand_signal = hilbert(padd_rand_signal')';
        analytic_rand_signal = analytic_rand_signal(:, n_pts+1:2*n_pts);

        f1 = analytic_rand_signal(:, 1:end-1);
        f2 = analytic_rand_signal(:, 2:end);
        amp_prods(i, :, :) = abs(f1) .* abs(f2);

    end