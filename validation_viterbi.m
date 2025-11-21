%% Paramètres
nb = 5000;                   % nombre de bits utiles
g = [7 5];                  % code convolutif (7,5)_8
trellis = poly2trellis(3, g);
m = log2(trellis.numStates);    % mémoire
ns = log2(trellis.numOutputSymbols);

EbN0dB = 0:0.5:7;
TEB_moi = zeros(size(EbN0dB));
nbIter = 100;                % nombre de trames par point SNR

%% Simulation TEB
for i = 1:length(EbN0dB)

    EbN0 = 10^(EbN0dB(i)/10);
    sigma2 = 1/(2 * EbN0);

    err = 0;
    total = 0;

    for it = 1:nbIter
        %% message utile
        u = randi([0 1], nb, 1);

        %% encodage 
        c = cc_encode(u, trellis);
        
        %% BPSK
        x = 1 - 2*c;
        
        %% canal
        y = x + sqrt(sigma2)*randn(size(x));
        
        %% LLR
        Lc = 2*y/sigma2;
        
        %% décodage 
        u_closed_est = viterbi_decode(Lc, trellis);
        u_est = u_closed_est(1:nb);   % retirer la fermeture
        
        %% erreurs
        err = err + sum(u ~= u_est);
        total = total + nb;
    end

    TEB_moi(i) = err/total;
    fprintf("Eb/N0=%g dB : TEB_moi = %.3e\n", EbN0dB(i), TEB_moi(i));
end

%% Affichage
figure; hold on;

% notre COURBE
semilogy(EbN0dB, TEB_moi, 'bo-', 'LineWidth', 2, 'DisplayName','Mon Viterbi');

% COURBE BERTOOL
%semilogy(ebno0, ber0, 'r--', 'LineWidth', 2, 'DisplayName','BERTOOL');

grid on;
xlabel('Eb/N0 (dB)');
ylabel('TEB');
legend('show');
title('Comparaison : Mon Viterbi vs BERTOOL');
ylim([1e-7 1]);
