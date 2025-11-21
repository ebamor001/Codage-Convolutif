function u_decode = viterbi_decode(Lc, trellis)
% Entrées :
%   Lc : un vecteur de ns.L observations du canal
%   trellis : structure du code convolutif
% Sortie :
%   u_decode : message estimé (binaire)

nbs=trellis.numStates; %nbre de states
m = log2(trellis.numStates);
ns = log2(trellis.numOutputSymbols);
L = length(Lc)/ns;

% Matrices pour les chemins
%JL(s,i)==meilleur chemin pour arriver à s à l'instant i (les +1 c psq les
%tabelaux en matlab commencent par 1)
JL = inf(nbs, L+1);     % stocke le coût cumulé minimal pour arriver à chaque état à chaque instant
JL(1,1) = 0;            % État initial = 0 et tout les autres chemins sont encore inattaignables donc on les initialise à l'infini
Precedent = zeros(nbs, L);    % Prédécesseurs stockant le meilleur chemin
bits = zeros(nbs, L);   % stocke le bit d'entrée ayant mené à ce meilleur chemin.

for i = 1:L
    for s = 0:nbs-1
        for input = 0:1
            s_prev = find(trellis.nextStates(:,input+1)==s)-1; %on stocke les états précédents possibles
            for k = 1:length(s_prev)
                %on garde que celui qui a le moins cout
                c_out = de2bi(trellis.outputs(s_prev(k)+1,input+1), ns, 'left-msb');
                c_out_bip = 1 - 2*c_out;      % map 0→+1, 1→-1
                y_seg = Lc((i-1)*ns+1:i*ns);
                metric = - sum(y_seg .* c_out_bip(:));
                new_metric = JL(s_prev(k)+1,i) + metric;
                if new_metric < JL(s+1,i+1)
                    JL(s+1,i+1) = new_metric;
                    Precedent(s+1,i) = s_prev(k);
                    bits(s+1,i) = input;
                end
            end
        end
    end
end

% Backtracking
state = 0;
u_decode = zeros(L,1);
for i = L:-1:1
    u_decode(i) = bits(state+1,i);
    state = Precedent(state+1,i);
end
u_decode = u_decode(1:end-m); % supprimer les bits de fermeture
end
