clear; clc; close all;

%% Paramètres
K = 1024;
M = 2;
EbN0dB = 0:1:8;
nbrErreur = 100;
nbrBitMax = 1e6;

% Les 4 encodeurs demandés
codes = { [5 7], [1 5 7], [13 15], [1 13 15] };
labels = {'(5 7)_8', '(1 5 7)_8 récursif', '(13 15)_8', '(1 13 15)_8 récursif'};

Ncodes = length(codes);
TEB = zeros(Ncodes, length(EbN0dB));

%% Simulation
for idx = 1:Ncodes

    g = codes{idx};
    m = find_memory(g);
    trellis = poly2trellis(m+1, g);

    fprintf("=== Code %s, mémoire = %d ===\n", labels{idx}, m);

    for iSNR = 1:length(EbN0dB)

        EbN0 = 10^(EbN0dB(iSNR)/10);
        sigma2 = 1/(2*EbN0);

        bitErr = 0;
        totalBits = 0;

        while (bitErr < nbrErreur && totalBits < nbrBitMax)

            u = randi([0 1], K, 1);
            c = cc_encode(u, trellis);

            x = 1 - 2*c;
            y = x + sqrt(sigma2)*randn(size(x));

            Lc = 2*y/sigma2;
            u_est_full = viterbi_decode(Lc, trellis);
            u_est = u_est_full(1:K);

            bitErr = bitErr + sum(u ~= u_est);
            totalBits = totalBits + K;
        end

        TEB(idx, iSNR) = bitErr/totalBits;
        fprintf("Eb/N0=%g dB -> TEB = %.3e\n", EbN0dB(iSNR), TEB(idx,iSNR));

    end
end

%% Affichage
figure; hold on;
colors = lines(Ncodes);

for idx = 1:Ncodes
    semilogy(EbN0dB, TEB(idx,:), 'o-', 'LineWidth', 2, ...
        'DisplayName', labels{idx});
end

grid on;
xlabel('Eb/N0 (dB)');
ylabel('TEB');
legend('show');
title('Comparaison RSC vs Non-récursif');
ylim([1e-7 1]);


function m = find_memory(g)
    max_len = 0;
    for i = 1:length(g)
        decimal_value = base2dec(num2str(g(i)), 8);
        b = de2bi(decimal_value); %conversion en binaire
        last1 = find(b,1,'last');  %% position du dernier '1'
        if last1 > max_len
            max_len = last1;
        end
    end
    m = max_len - 1;
end
