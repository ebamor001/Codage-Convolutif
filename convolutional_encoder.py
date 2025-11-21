"""
Encodeur convolutif
Module pour encoder des données binaires avec un code convolutif
"""
import numpy as np


class ConvolutionalEncoder:
    """
    Encodeur de code convolutif
    
    Paramètres:
    -----------
    constraint_length : int
        Longueur de contrainte K (nombre de bits dans le registre à décalage)
    generator_polynomials : list of int
        Polynômes générateurs en notation octale (par exemple [7, 5] pour rate 1/2)
    """
    
    def __init__(self, constraint_length, generator_polynomials):
        self.K = constraint_length
        self.generators = generator_polynomials
        self.n = len(generator_polynomials)  # Nombre de bits de sortie par bit d'entrée
        self.rate = 1.0 / self.n  # Taux de codage
        self.memory = self.K - 1  # Longueur de la mémoire
        
    def _apply_generator_polynomial(self, register, generator):
        """
        Applique un polynôme générateur au registre
        
        Paramètres:
        -----------
        register : numpy array
            Registre à décalage
        generator : int
            Polynôme générateur en notation octale
            
        Retourne:
        ---------
        output_bit : int
            Bit de sortie
        """
        output_bit = 0
        gen_binary = bin(generator)[2:][::-1]  # Convertir en binaire et inverser
        for i, g_bit in enumerate(gen_binary):
            if i < self.K and g_bit == '1':
                output_bit ^= register[i]
        return output_bit
    
    def encode(self, input_bits):
        """
        Encode une séquence de bits
        
        Paramètres:
        -----------
        input_bits : array-like
            Séquence de bits d'entrée (0 ou 1)
            
        Retourne:
        ---------
        output_bits : numpy array
            Séquence de bits encodés
        """
        input_bits = np.array(input_bits, dtype=int)
        n_bits = len(input_bits)
        
        # Initialiser le registre à décalage avec des zéros
        register = np.zeros(self.K, dtype=int)
        
        # Stocker les bits de sortie
        output_bits = []
        
        # Encoder chaque bit
        for bit in input_bits:
            # Décaler et insérer le nouveau bit
            register = np.roll(register, 1)
            register[0] = bit
            
            # Calculer les bits de sortie pour chaque polynôme générateur
            for gen in self.generators:
                output_bit = self._apply_generator_polynomial(register, gen)
                output_bits.append(output_bit)
        
        # Ajouter les bits de terminaison (flush)
        for _ in range(self.memory):
            register = np.roll(register, 1)
            register[0] = 0
            
            for gen in self.generators:
                output_bit = self._apply_generator_polynomial(register, gen)
                output_bits.append(output_bit)
        
        return np.array(output_bits, dtype=int)
    
    def get_num_states(self):
        """Retourne le nombre d'états dans le treillis"""
        return 2 ** self.memory
    
    def get_next_state(self, current_state, input_bit):
        """
        Calcule l'état suivant et la sortie
        
        Paramètres:
        -----------
        current_state : int
            État actuel (0 à 2^memory - 1)
        input_bit : int
            Bit d'entrée (0 ou 1)
            
        Retourne:
        ---------
        next_state : int
            État suivant
        output : list
            Bits de sortie
        """
        # Reconstruire le registre à partir de l'état
        register = np.zeros(self.K, dtype=int)
        state_bits = format(current_state, f'0{self.memory}b')
        for i, bit in enumerate(state_bits):
            register[i + 1] = int(bit)
        register[0] = input_bit
        
        # Calculer la sortie
        output = []
        for gen in self.generators:
            output_bit = self._apply_generator_polynomial(register, gen)
            output.append(output_bit)
        
        # Calculer l'état suivant
        next_state = (current_state >> 1) | (input_bit << (self.memory - 1))
        
        return next_state, output
