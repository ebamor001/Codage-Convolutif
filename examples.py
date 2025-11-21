"""
Exemple simple d'utilisation des codes convolutifs
"""
import numpy as np
from convolutional_encoder import ConvolutionalEncoder
from viterbi_decoder import ViterbiDecoder
from simulator import ConvolutionalCodeSimulator


def example_basic():
    """Exemple de base : encodage et décodage sans bruit"""
    print("=" * 60)
    print("EXEMPLE 1 : Encodage et décodage sans bruit")
    print("=" * 60)
    
    # Créer un encodeur rate 1/2, K=3
    # Polynômes générateurs : [7, 5] en octal = [111, 101] en binaire
    encoder = ConvolutionalEncoder(constraint_length=3, generator_polynomials=[7, 5])
    print(f"\nConfiguration de l'encodeur:")
    print(f"  - Longueur de contrainte K = {encoder.K}")
    print(f"  - Polynômes générateurs = {encoder.generators} (octal)")
    print(f"  - Taux de codage = {encoder.rate}")
    print(f"  - Nombre d'états = {encoder.get_num_states()}")
    
    # Données à encoder
    data_bits = np.array([1, 0, 1, 1, 0])
    print(f"\nBits de données : {data_bits}")
    
    # Encoder
    encoded_bits = encoder.encode(data_bits)
    print(f"Bits encodés    : {encoded_bits}")
    print(f"Longueur        : {len(data_bits)} bits -> {len(encoded_bits)} bits")
    
    # Créer un décodeur
    decoder = ViterbiDecoder(encoder)
    
    # Décoder (sans bruit)
    decoded_bits = decoder.decode(encoded_bits)
    print(f"Bits décodés    : {decoded_bits}")
    
    # Vérifier
    if np.array_equal(data_bits, decoded_bits):
        print("\n✓ Décodage réussi ! Les bits sont identiques.")
    else:
        print("\n✗ Erreur de décodage !")
    

def example_with_noise():
    """Exemple avec bruit"""
    print("\n\n" + "=" * 60)
    print("EXEMPLE 2 : Transmission avec bruit")
    print("=" * 60)
    
    # Créer l'encodeur et le décodeur
    encoder = ConvolutionalEncoder(constraint_length=3, generator_polynomials=[7, 5])
    decoder = ViterbiDecoder(encoder)
    simulator = ConvolutionalCodeSimulator(encoder, decoder)
    
    # Données à transmettre
    data_bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    print(f"\nBits de données : {data_bits}")
    
    # Simuler avec différents niveaux de bruit
    snr_values = [0, 3, 6]
    
    for snr_db in snr_values:
        print(f"\n--- SNR = {snr_db} dB ---")
        
        # Simulation avec décodage souple
        results = simulator.simulate_single_transmission(data_bits, snr_db, soft_decision=True)
        
        print(f"Bits encodés     : {results['encoded_bits']}")
        print(f"Bits décodés     : {results['decoded_bits']}")
        print(f"Erreurs de bits  : {results['bit_errors']}")
        print(f"Taux d'erreur    : {results['ber']:.4f}")


def example_ber_curve():
    """Exemple de courbe BER"""
    print("\n\n" + "=" * 60)
    print("EXEMPLE 3 : Simulation de courbe BER")
    print("=" * 60)
    
    # Créer l'encodeur et le décodeur
    encoder = ConvolutionalEncoder(constraint_length=3, generator_polynomials=[7, 5])
    decoder = ViterbiDecoder(encoder)
    simulator = ConvolutionalCodeSimulator(encoder, decoder)
    
    # Paramètres de simulation
    block_length = 100
    snr_range = np.arange(0, 7, 2)
    num_blocks = 50
    
    print(f"\nParamètres de simulation:")
    print(f"  - Longueur de bloc : {block_length} bits")
    print(f"  - Plage SNR        : {snr_range[0]} à {snr_range[-1]} dB")
    print(f"  - Nombre de blocs  : {num_blocks}")
    
    print("\nSimulation en cours...")
    results = simulator.simulate_ber(block_length, snr_range, num_blocks, soft_decision=True)
    
    print("\nRésultats:")
    print(f"{'SNR (dB)':<10} {'BER Codé':<15} {'BER Non-codé':<15}")
    print("-" * 40)
    for i, snr in enumerate(results['snr_db']):
        print(f"{snr:<10.1f} {results['ber_coded'][i]:<15.6f} {results['ber_uncoded'][i]:<15.6f}")


def example_different_codes():
    """Exemple avec différents codes"""
    print("\n\n" + "=" * 60)
    print("EXEMPLE 4 : Différents codes convolutifs")
    print("=" * 60)
    
    # Différentes configurations
    configs = [
        {"K": 3, "gen": [7, 5], "name": "Rate 1/2, K=3 (NASA)"},
        {"K": 3, "gen": [7, 5, 3], "name": "Rate 1/3, K=3"},
        {"K": 4, "gen": [15, 17], "name": "Rate 1/2, K=4"},
    ]
    
    data_bits = np.random.randint(0, 2, 50)
    snr_db = 3
    
    print(f"\nBits de données : {len(data_bits)} bits")
    print(f"SNR             : {snr_db} dB\n")
    
    for config in configs:
        print(f"\n{config['name']}")
        print("-" * 40)
        
        encoder = ConvolutionalEncoder(config['K'], config['gen'])
        decoder = ViterbiDecoder(encoder)
        simulator = ConvolutionalCodeSimulator(encoder, decoder)
        
        results = simulator.simulate_single_transmission(data_bits, snr_db, soft_decision=True)
        
        print(f"  Taux de codage : {encoder.rate:.3f}")
        print(f"  Bits encodés   : {len(results['encoded_bits'])} bits")
        print(f"  Erreurs        : {results['bit_errors']} / {len(data_bits)}")
        print(f"  BER            : {results['ber']:.6f}")


if __name__ == "__main__":
    np.random.seed(42)  # Pour la reproductibilité
    
    example_basic()
    example_with_noise()
    example_ber_curve()
    example_different_codes()
    
    print("\n" + "=" * 60)
    print("Exemples terminés !")
    print("=" * 60)
