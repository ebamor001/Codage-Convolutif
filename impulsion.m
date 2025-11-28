clear; clc; close all;

%% Charger résultat Monte-Carlo du code (5,7)8
load('Code_5  7.mat');   % doit contenir EbN0dB et TEP

TEP_simule = TEP(1,:);    % première ligne = code (5,7)
EbN0dB_sim = EbN0dB;       % renommage pour clarté

%% Paramètres impulsion
g = [5 7];
trellis = poly2trellis(3, g);
K = 1024;                  % même K que la simulation
d0 = 1;
d1 = 100;

%% Méthode Impulsion
[~, TEP_impulsion] = impulse_method(trellis, K, EbN0dB_sim, d0, d1);

%% Plot
figure;
semilogy(EbN0dB_sim, TEP_impulsion, 'r-', 'LineWidth', 2); hold on;
semilogy(EbN0dB_sim, TEP_simule, 'bo-', 'LineWidth', 2);

grid on;
legend('TEP Impulsion', 'Monte-Carlo');
title('Comparaison TEP : Méthode Impulsion vs Simulation Monte-Carlo');
xlabel('Eb/N0 (dB)');
ylabel('TEP');




function [EbN0dB, TEP_estime] = impulse_method(trellis, K, EbN0dB, d0, d1)
    m = log2(trellis.numStates);
    ns = log2(trellis.numOutputSymbols);
    N = (K+m) * ns;   % longueur du mot codé
    
    v = zeros(1, K);
    xu = zeros(K,1);               
    y_ref = ones(N,1);               % observations parfaites 
   
    
    for l = 1:K
        A = d0 - 0.5;
        xu_ref = xu;    % séquence correcte
        xu_est = xu_ref;
        pos = (l-1)*ns + 1;
        while (all(xu_est == xu_ref) && A <= d1)
            A = A + 1;
    
            % construire y perturbé
            y = y_ref;     

            % injecter l'impulsion
            y(pos) = 1 - A;
    
            xu_est = viterbi_decode(y, trellis);
    
        end
    
        v(l) = A;    % distance impulsionnelle du bit l
    
    end
    
    %% 5) Construire D et A_d
    D = unique(v);
    Ad = arrayfun(@(d) sum(v == d), D);
    
    %% 6) Calcul du TEP
    R = 1/ns;
    TEP_estime = zeros(size(EbN0dB));
    
    for i = 1:length(EbN0dB)
        EbN0 = 10^(EbN0dB(i)/10);
    
        somme = 0;
        for k = 1:length(D)
            d = D(k);
            somme = somme + Ad(k) * erfc( sqrt( d * R * EbN0 ) );
        end
    
        TEP_estime(i) = 0.5 * somme;
    end

end