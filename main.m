clear;
clc;
close all;

%% Paramètres généraux
K = 1024;              % Nombre de bits du message
N = 2048;              % Nombre de bits codés par trame 
R = K/N;               % Rendement de la communication

M = 2;                 % Modulation BPSK <=> 2 symboles

EbN0dB_min  = 0;       % Eb/N0 min
EbN0dB_max  = 10;      % Eb/N0 max
EbN0dB_step = 1;       % Pas
nbrErreur   = 100;     % Nombre d'erreurs à observer avant de calculer un BER
nbrBitMax   = 1e6;     % Nb max de bits simulés
TEBMin      = 3e-6;    % BER minimal visé

EbN0dB  = EbN0dB_min:EbN0dB_step:EbN0dB_max;
EbN0    = 10.^(EbN0dB/10);
EsN0    = R*log2(M)*EbN0; % Points de EsN0 =EbN0 pour BPSK
sigmaz2 = 1./(2 * EsN0);  % Variance de bruit pour chaque EbN0

% Codes à étudier (octal)
%codes = { [2 3], [5 7], [13 15], [133 171] };
codes = { [5 7], [1 5 7], [13 15], [1 13 15] };

Ncodes = length(codes);

% Initialisation des vecteurs de résultats
TEB = zeros(Ncodes, length(EbN0dB));
TEP = zeros(Ncodes, length(EbN0dB)); %packet error rate

Pb_u = qfunc(sqrt(2*EbN0)); % Probabilité d'erreur non codée
Pe_u = 1 - (1 - Pb_u).^K;

%% Préparation affichage
figure(1)
semilogy(EbN0dB, Pb_u, '--', 'LineWidth',1.5,'DisplayName','Pb (BPSK théorique)');
hold all
semilogy(EbN0dB, Pe_u, '--', 'LineWidth',1.5,'DisplayName','Pe (BPSK théorique)');
grid on
xlabel('$\frac{E_b}{N_0}$ en dB','Interpreter','latex','FontSize',14)
ylabel('TEB / TEP','Interpreter','latex','FontSize',14)
ylim([1e-6 1])
legend()

%% Préparation de l'affichage en console
line       =  '|------------|---------------|------------|------------|----------|----------|------------------|-------------------|--------------|\n';
msg_header =  '|  Eb/N0 dB  |    Bit nbr    |  Bit err   |  Pqt err   |   TEB    |   TEP    |     Debit Tx     |      Debit Rx     | Tps restant  |\n';
msgFormat  =  '|   %7.2f  |   %9d   |  %9d |  %9d | %2.2e | %2.2e |  %10.2f MO/s |   %10.2f MO/s |   %8.2f s |\n';

%% Boucle sur les différents codes
for idx = 1:Ncodes
    g = codes{idx};
    m = find_memory(g);
    trellis = poly2trellis(m+1, g);
    ns = log2(trellis.numOutputSymbols);
    R_c = 1/ns;

    fprintf('\n=============================================================\n');
    fprintf('Simulation du code (%s)_8, mémoire = %d, taux R = %.2f\n', num2str(g), m, R_c);
    fprintf('=============================================================\n');

    % Initialisation affichage console
    fprintf(line);
    fprintf(msg_header);
    fprintf(line);

    % Préparation tracé
    hTEB = semilogy(EbN0dB,TEB(idx,:), 'LineWidth',1.5,'DisplayName',sprintf('TEB (%s)_8', num2str(g)));
    hTEP = semilogy(EbN0dB,TEP(idx,:), 'LineWidth',1.5,'DisplayName',sprintf('TEP (%s)_8', num2str(g)));

    for iSNR = 1:length(EbN0dB)
        reverseStr = '';% Pour affichage en console stat_erreur
        pqtNbr = 0;     % Nombre de paquets envoyés
        bitErr = 0;     % Nombre de bits faux
        pqtErr = 0;     % Nombre de paquets faux
        T_rx = 0;
        T_tx = 0;
        general_tic = tic;

        while (bitErr < nbrErreur && pqtNbr*K < nbrBitMax)
            pqtNbr = pqtNbr + 1;

            %% Émetteur
            tx_tic = tic;               % Mesure du débit d'encodage
            u = randi([0 1], K, 1);     % Génération du message aléatoire
            c = cc_encode(u, trellis);  % Encodage
            x = 1 - 2*c;                % Modulation BPSK
            T_tx = T_tx + toc(tx_tic);  % Mesure du débit d'encodage
            debitTX = pqtNbr*K/8/T_tx/1e6;

            %% Canal AWGN
            z = sqrt(sigmaz2(iSNR)) * randn(size(x));  % Génération du bruit blanc gaussien
            y = x + z;              % Ajout du bruit blanc gaussien

            %% Récepteur
            rx_tic = tic;
            Lc = (2/sigmaz2(iSNR)) * y;   % LLR du canal
            u_rec = viterbi_decode(Lc, trellis);
            T_rx = T_rx + toc(rx_tic);
            debitRX = pqtNbr*K/8/T_rx/1e6;

            %% Calculs des erreurs
            BE = sum(u(:) ~= u_rec(:));
            bitErr = bitErr + BE;
            pqtErr = pqtErr + double(BE > 0);

            %% Affichage dynamique
            if mod(pqtNbr,100) == 1
                pct1 = bitErr/nbrErreur;
                pct2 = pqtNbr*K/nbrBitMax;
                pct = max(pct1, pct2);

                display_str = sprintf(msgFormat,...
                    EbN0dB(iSNR),               ... % EbN0 en dB
                    pqtNbr*K,                   ... % Nombre de bits envoyés
                    bitErr,                     ... % Nombre d'erreurs observées
                    pqtErr,                     ... % Nombre d'erreurs observées
                    bitErr/(pqtNbr*K),          ... % TEB
                    pqtErr/pqtNbr,              ... % TEP
                    debitTX,                    ... % Débit d'encodage
                    debitRX,                    ... % Débit de décodage
                    toc(general_tic)/pct*(1-pct)); % Temps restant
                
                lr = length(reverseStr);
                msg_sz = fprintf([reverseStr, display_str]);
                reverseStr = repmat(sprintf('\b'),1,msg_sz-lr);

                TEB(idx,iSNR) = bitErr/(pqtNbr*K);
                TEP(idx,iSNR) = pqtErr/pqtNbr;
                set(hTEB, 'YData', TEB(idx,:));
                set(hTEP, 'YData', TEP(idx,:));
                drawnow;

            end
        end

        display_str = sprintf(msgFormat, EbN0dB(iSNR), pqtNbr*K, bitErr, pqtErr,...
                              bitErr/(pqtNbr*K), pqtErr/pqtNbr, debitTX, debitRX, 0);
        fprintf(reverseStr);
        msg_sz = fprintf(display_str);
        reverseStr = repmat(sprintf('\b'), 1, msg_sz);

        TEB(idx,iSNR) = bitErr/(pqtNbr*K);
        TEP(idx,iSNR) = pqtErr/pqtNbr;
        refreshdata(hTEB);
        refreshdata(hTEP);
        drawnow limitrate;

        if TEB(idx,iSNR) < TEBMin
            break;
        end
    end

    fprintf(line);
    save(sprintf('Code_%s.mat', num2str(g)), 'EbN0dB','TEB','TEP','R_c','K','trellis');
end

%% Affichage final
grid on;
xlabel('$\frac{E_b}{N_0}$ en dB','Interpreter', 'latex', 'FontSize',14)
ylabel('TEB / TEP','Interpreter', 'latex', 'FontSize',14)
legend('show', 'Interpreter', 'latex', 'FontSize',12);
ylim([1e-6 1])
xlim([0 12])

fprintf('\nSimulation terminée ✅\n');


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
