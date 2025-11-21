function c = cc_encode(u, trellis)
% cc_encode : encodeur convolutif binaire
% Entrées :
%   u : message binaire [Kxnb]
%   trellis : structure du treillis (poly2trellis)
% Sortie :
%   c : mot de code binaire

m = log2(trellis.numStates);  % mémoire
ns=log2(trellis.numOutputSymbols);  %taille d'un symbole de sortie
state = 0;                    % état initial
L = length(u) + m;            % fermeture du treillis
u_ferme = [u; zeros(m,1)];      % fermeture de taille L

c = zeros(L * ns,1); 
for i = 1:L
    input = u_ferme(i);
    output = de2bi(trellis.outputs(state+1, input+1), ns, 'left-msb'); %bit de poids fort à gauche 
    c((i-1)*ns+1:i*ns) = output(:);
    state = trellis.nextStates(state+1, input+1);
end
end
