"""
Décodeur de Viterbi
Module pour décoder les codes convolutifs avec l'algorithme de Viterbi
"""
import numpy as np


class ViterbiDecoder:
    """
    Décodeur de Viterbi pour codes convolutifs
    
    Paramètres:
    -----------
    encoder : ConvolutionalEncoder
        Instance de l'encodeur convolutif correspondant
    """
    
    def __init__(self, encoder):
        self.encoder = encoder
        self.num_states = encoder.get_num_states()
        
    def hamming_distance(self, seq1, seq2):
        """Calcule la distance de Hamming entre deux séquences"""
        return np.sum(np.array(seq1) != np.array(seq2))
    
    def _reconstruct_bits_from_path(self, path):
        """
        Reconstruit les bits d'entrée à partir d'un chemin d'états
        
        Paramètres:
        -----------
        path : list
            Séquence d'états
            
        Retourne:
        ---------
        decoded_bits : list
            Séquence de bits décodés
        """
        decoded_bits = []
        for i in range(len(path) - 1):
            current_state = path[i]
            next_state = path[i + 1]
            
            # Déterminer quel bit d'entrée a causé cette transition
            bit_found = False
            for input_bit in [0, 1]:
                test_next_state, _ = self.encoder.get_next_state(current_state, input_bit)
                if test_next_state == next_state:
                    decoded_bits.append(input_bit)
                    bit_found = True
                    break
            
            # Ceci ne devrait jamais arriver avec un chemin valide
            if not bit_found:
                # En cas d'erreur, ajouter 0 par défaut
                decoded_bits.append(0)
        
        # Retirer les bits de terminaison
        if len(decoded_bits) > self.encoder.memory:
            decoded_bits = decoded_bits[:-self.encoder.memory]
        
        return decoded_bits
    
    def decode(self, received_bits):
        """
        Décode une séquence reçue avec l'algorithme de Viterbi
        
        Paramètres:
        -----------
        received_bits : array-like
            Séquence de bits reçus (potentiellement bruités)
            
        Retourne:
        ---------
        decoded_bits : numpy array
            Séquence de bits décodés
        """
        received_bits = np.array(received_bits, dtype=int)
        n = self.encoder.n
        
        # Nombre de pas de temps
        num_steps = len(received_bits) // n
        
        # Initialiser les métriques de chemin
        path_metrics = np.full(self.num_states, np.inf)
        path_metrics[0] = 0  # Commencer à l'état 0
        
        # Stocker les chemins
        paths = [[i] for i in range(self.num_states)]
        
        # Algorithme de Viterbi
        for t in range(num_steps):
            # Extraire les bits reçus pour ce pas de temps
            received_symbol = received_bits[t * n:(t + 1) * n]
            
            # Nouvelles métriques et chemins
            new_path_metrics = np.full(self.num_states, np.inf)
            new_paths = [None] * self.num_states
            
            # Pour chaque état actuel
            for current_state in range(self.num_states):
                if path_metrics[current_state] == np.inf:
                    continue
                
                # Essayer les deux entrées possibles (0 et 1)
                for input_bit in [0, 1]:
                    next_state, output = self.encoder.get_next_state(current_state, input_bit)
                    
                    # Calculer la métrique de branche (distance de Hamming)
                    branch_metric = self.hamming_distance(output, received_symbol)
                    new_metric = path_metrics[current_state] + branch_metric
                    
                    # Mettre à jour si c'est un meilleur chemin
                    if new_metric < new_path_metrics[next_state]:
                        new_path_metrics[next_state] = new_metric
                        new_paths[next_state] = paths[current_state] + [next_state]
            
            path_metrics = new_path_metrics
            paths = new_paths
        
        # Trouver le meilleur chemin final (devrait se terminer à l'état 0)
        # Si terminaison forcée, choisir l'état 0
        if path_metrics[0] != np.inf:
            best_state = 0
        else:
            # Sinon, choisir l'état avec la meilleure métrique
            best_state = np.argmin(path_metrics)
        
        best_path = paths[best_state]
        
        # Reconstruire les bits d'entrée à partir du chemin
        decoded_bits = self._reconstruct_bits_from_path(best_path)
        
        return np.array(decoded_bits, dtype=int)
    
    def decode_soft(self, received_symbols):
        """
        Décode avec des décisions souples (soft decision)
        
        Paramètres:
        -----------
        received_symbols : array-like
            Symboles reçus (valeurs réelles, pas nécessairement binaires)
            
        Retourne:
        ---------
        decoded_bits : numpy array
            Séquence de bits décodés
        """
        received_symbols = np.array(received_symbols, dtype=float)
        n = self.encoder.n
        
        # Nombre de pas de temps
        num_steps = len(received_symbols) // n
        
        # Initialiser les métriques de chemin
        path_metrics = np.full(self.num_states, np.inf)
        path_metrics[0] = 0
        
        # Stocker les chemins
        paths = [[i] for i in range(self.num_states)]
        
        # Algorithme de Viterbi avec décisions souples
        for t in range(num_steps):
            received_symbol = received_symbols[t * n:(t + 1) * n]
            
            new_path_metrics = np.full(self.num_states, np.inf)
            new_paths = [None] * self.num_states
            
            for current_state in range(self.num_states):
                if path_metrics[current_state] == np.inf:
                    continue
                
                for input_bit in [0, 1]:
                    next_state, output = self.encoder.get_next_state(current_state, input_bit)
                    
                    # Calculer la métrique euclidienne
                    expected = np.array(output, dtype=float) * 2 - 1  # Mapper 0->-1, 1->1
                    branch_metric = np.sum((received_symbol - expected) ** 2)
                    new_metric = path_metrics[current_state] + branch_metric
                    
                    if new_metric < new_path_metrics[next_state]:
                        new_path_metrics[next_state] = new_metric
                        new_paths[next_state] = paths[current_state] + [next_state]
            
            path_metrics = new_path_metrics
            paths = new_paths
        
        # Trouver le meilleur chemin
        if path_metrics[0] != np.inf:
            best_state = 0
        else:
            best_state = np.argmin(path_metrics)
        
        best_path = paths[best_state]
        
        # Reconstruire les bits
        decoded_bits = self._reconstruct_bits_from_path(best_path)
        
        return np.array(decoded_bits, dtype=int)
