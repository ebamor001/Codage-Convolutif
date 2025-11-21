"""
Simulateur pour codes convolutifs
Module pour simuler des transmissions avec bruit et mesurer les performances
"""
import numpy as np
from convolutional_encoder import ConvolutionalEncoder
from viterbi_decoder import ViterbiDecoder


class ConvolutionalCodeSimulator:
    """
    Simulateur pour codes convolutifs avec canal bruité
    
    Paramètres:
    -----------
    encoder : ConvolutionalEncoder
        Encodeur convolutif
    decoder : ViterbiDecoder
        Décodeur de Viterbi
    """
    
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
    
    def _count_bit_errors(self, original_bits, decoded_bits):
        """
        Compte les erreurs de bits entre deux séquences
        
        Paramètres:
        -----------
        original_bits : array-like
            Bits originaux
        decoded_bits : array-like
            Bits décodés
            
        Retourne:
        ---------
        errors : int
            Nombre d'erreurs
        min_len : int
            Longueur minimale comparée
        """
        min_len = min(len(original_bits), len(decoded_bits))
        errors = np.sum(original_bits[:min_len] != decoded_bits[:min_len])
        return errors, min_len
        
    def add_awgn_noise(self, signal, snr_db):
        """
        Ajoute du bruit blanc gaussien additif (AWGN)
        
        Paramètres:
        -----------
        signal : array-like
            Signal à bruiter (bits 0/1)
        snr_db : float
            Rapport signal sur bruit en dB
            
        Retourne:
        ---------
        noisy_signal : numpy array
            Signal bruité (valeurs réelles)
        """
        signal = np.array(signal, dtype=float)
        
        # Convertir les bits en symboles BPSK: 0 -> -1, 1 -> +1
        symbols = 2 * signal - 1
        
        # Calculer la puissance du bruit
        signal_power = np.mean(symbols ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        # Générer et ajouter le bruit
        noise = np.random.normal(0, np.sqrt(noise_power), len(symbols))
        noisy_signal = symbols + noise
        
        return noisy_signal
    
    def add_bsc_noise(self, bits, error_probability):
        """
        Simule un canal binaire symétrique (BSC)
        
        Paramètres:
        -----------
        bits : array-like
            Bits à transmettre
        error_probability : float
            Probabilité d'erreur par bit
            
        Retourne:
        ---------
        noisy_bits : numpy array
            Bits après passage dans le canal
        """
        bits = np.array(bits, dtype=int)
        
        # Générer des erreurs aléatoires
        errors = np.random.random(len(bits)) < error_probability
        noisy_bits = (bits + errors) % 2
        
        return noisy_bits
    
    def simulate_ber(self, block_length, snr_db_range, num_blocks=100, soft_decision=True):
        """
        Simule le taux d'erreur binaire (BER) pour différents SNR
        
        Paramètres:
        -----------
        block_length : int
            Longueur des blocs de données
        snr_db_range : array-like
            Plage de valeurs SNR en dB
        num_blocks : int
            Nombre de blocs à simuler pour chaque SNR
        soft_decision : bool
            Utiliser le décodage souple (True) ou dur (False)
            
        Retourne:
        ---------
        ber_results : dict
            Dictionnaire avec les résultats de simulation
        """
        snr_db_range = np.array(snr_db_range)
        ber_coded = []
        ber_uncoded = []
        
        for snr_db in snr_db_range:
            total_errors_coded = 0
            total_errors_uncoded = 0
            total_bits_coded = 0
            total_bits_uncoded = 0
            
            for _ in range(num_blocks):
                # Générer des bits aléatoires
                data_bits = np.random.randint(0, 2, block_length)
                
                # Encoder
                encoded_bits = self.encoder.encode(data_bits)
                
                # Ajouter du bruit AWGN
                noisy_signal = self.add_awgn_noise(encoded_bits, snr_db)
                
                # Décoder
                if soft_decision:
                    decoded_bits = self.decoder.decode_soft(noisy_signal)
                else:
                    # Décision dure
                    hard_bits = (noisy_signal > 0).astype(int)
                    decoded_bits = self.decoder.decode(hard_bits)
                
                # Compter les erreurs (codé)
                errors_coded, min_len = self._count_bit_errors(data_bits, decoded_bits)
                total_errors_coded += errors_coded
                total_bits_coded += min_len
                
                # Erreurs non codées (pour comparaison)
                noisy_data = self.add_awgn_noise(data_bits, snr_db)
                uncoded_bits = (noisy_data > 0).astype(int)
                errors_uncoded = np.sum(data_bits != uncoded_bits)
                total_errors_uncoded += errors_uncoded
                total_bits_uncoded += len(data_bits)
            
            # Calculer les BER
            ber_coded.append(total_errors_coded / total_bits_coded if total_bits_coded > 0 else 0)
            ber_uncoded.append(total_errors_uncoded / total_bits_uncoded if total_bits_uncoded > 0 else 0)
        
        return {
            'snr_db': snr_db_range,
            'ber_coded': np.array(ber_coded),
            'ber_uncoded': np.array(ber_uncoded),
            'coding_gain_db': None  # À calculer si nécessaire
        }
    
    def simulate_single_transmission(self, data_bits, snr_db, soft_decision=True):
        """
        Simule une transmission unique
        
        Paramètres:
        -----------
        data_bits : array-like
            Bits de données à transmettre
        snr_db : float
            Rapport signal sur bruit en dB
        soft_decision : bool
            Utiliser le décodage souple
            
        Retourne:
        ---------
        results : dict
            Résultats détaillés de la transmission
        """
        data_bits = np.array(data_bits, dtype=int)
        
        # Encoder
        encoded_bits = self.encoder.encode(data_bits)
        
        # Ajouter du bruit
        noisy_signal = self.add_awgn_noise(encoded_bits, snr_db)
        
        # Décoder
        if soft_decision:
            decoded_bits = self.decoder.decode_soft(noisy_signal)
        else:
            hard_bits = (noisy_signal > 0).astype(int)
            decoded_bits = self.decoder.decode(hard_bits)
        
        # Calculer les erreurs
        bit_errors, min_len = self._count_bit_errors(data_bits, decoded_bits)
        ber = bit_errors / min_len if min_len > 0 else 0
        
        return {
            'data_bits': data_bits,
            'encoded_bits': encoded_bits,
            'noisy_signal': noisy_signal,
            'decoded_bits': decoded_bits,
            'bit_errors': bit_errors,
            'ber': ber
        }
