"""
Script de visualisation des performances
Génère des graphiques pour analyser les performances des codes convolutifs
"""
import numpy as np
import matplotlib.pyplot as plt
from convolutional_encoder import ConvolutionalEncoder
from viterbi_decoder import ViterbiDecoder
from simulator import ConvolutionalCodeSimulator


def plot_ber_curves():
    """Génère des courbes BER pour différentes configurations"""
    
    # Configuration de base
    encoder = ConvolutionalEncoder(constraint_length=3, generator_polynomials=[7, 5])
    decoder = ViterbiDecoder(encoder)
    simulator = ConvolutionalCodeSimulator(encoder, decoder)
    
    # Paramètres de simulation
    block_length = 200
    snr_range = np.arange(-2, 8, 1.0)
    num_blocks = 100
    
    print("Simulation en cours...")
    print(f"Longueur de bloc : {block_length} bits")
    print(f"Nombre de blocs  : {num_blocks}")
    print(f"Plage SNR        : {snr_range[0]} à {snr_range[-1]} dB")
    
    # Simuler
    results = simulator.simulate_ber(block_length, snr_range, num_blocks, soft_decision=True)
    
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    plt.semilogy(results['snr_db'], results['ber_coded'], 'b-o', label='Codé (Viterbi)', linewidth=2)
    plt.semilogy(results['snr_db'], results['ber_uncoded'], 'r--s', label='Non codé', linewidth=2)
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Taux d\'erreur binaire (BER)', fontsize=12)
    plt.title('Performance du code convolutif (K=3, rate 1/2)', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim([snr_range[0], snr_range[-1]])
    
    plt.tight_layout()
    plt.savefig('ber_curve.png', dpi=150)
    print("\nGraphique sauvegardé : ber_curve.png")
    
    # Afficher les résultats numériques
    print("\nRésultats détaillés:")
    print(f"{'SNR (dB)':<10} {'BER Codé':<15} {'BER Non-codé':<15}")
    print("-" * 40)
    for i, snr in enumerate(results['snr_db']):
        print(f"{snr:<10.1f} {results['ber_coded'][i]:<15.6e} {results['ber_uncoded'][i]:<15.6e}")


def compare_different_codes():
    """Compare différents codes convolutifs"""
    
    configs = [
        {"K": 3, "gen": [7, 5], "name": "K=3, rate 1/2", "marker": 'o'},
        {"K": 3, "gen": [7, 5, 3], "name": "K=3, rate 1/3", "marker": 's'},
        {"K": 4, "gen": [15, 17], "name": "K=4, rate 1/2", "marker": '^'},
    ]
    
    block_length = 150
    snr_range = np.arange(0, 8, 1.5)
    num_blocks = 80
    
    print("\nComparaison de différents codes...")
    
    plt.figure(figsize=(10, 6))
    
    for config in configs:
        print(f"\nSimulation : {config['name']}")
        
        encoder = ConvolutionalEncoder(config['K'], config['gen'])
        decoder = ViterbiDecoder(encoder)
        simulator = ConvolutionalCodeSimulator(encoder, decoder)
        
        results = simulator.simulate_ber(block_length, snr_range, num_blocks, soft_decision=True)
        
        plt.semilogy(results['snr_db'], results['ber_coded'], 
                    marker=config['marker'], label=config['name'], linewidth=2)
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Taux d\'erreur binaire (BER)', fontsize=12)
    plt.title('Comparaison de différents codes convolutifs', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('comparison_codes.png', dpi=150)
    print("\nGraphique sauvegardé : comparison_codes.png")


def plot_soft_vs_hard_decision():
    """Compare le décodage souple et dur"""
    
    encoder = ConvolutionalEncoder(constraint_length=3, generator_polynomials=[7, 5])
    decoder = ViterbiDecoder(encoder)
    simulator = ConvolutionalCodeSimulator(encoder, decoder)
    
    block_length = 200
    snr_range = np.arange(0, 8, 1.0)
    num_blocks = 100
    
    print("\nComparaison décodage souple vs dur...")
    
    # Décodage souple
    print("Simulation avec décodage souple...")
    results_soft = simulator.simulate_ber(block_length, snr_range, num_blocks, soft_decision=True)
    
    # Décodage dur
    print("Simulation avec décodage dur...")
    results_hard = simulator.simulate_ber(block_length, snr_range, num_blocks, soft_decision=False)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(results_soft['snr_db'], results_soft['ber_coded'], 
                'b-o', label='Décodage souple', linewidth=2)
    plt.semilogy(results_hard['snr_db'], results_hard['ber_coded'], 
                'r--s', label='Décodage dur', linewidth=2)
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Taux d\'erreur binaire (BER)', fontsize=12)
    plt.title('Décodage souple vs dur (K=3, rate 1/2)', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('soft_vs_hard.png', dpi=150)
    print("\nGraphique sauvegardé : soft_vs_hard.png")


if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 60)
    print("GÉNÉRATION DES GRAPHIQUES DE PERFORMANCE")
    print("=" * 60)
    
    # Courbe BER de base
    plot_ber_curves()
    
    # Comparaison de différents codes
    compare_different_codes()
    
    # Décodage souple vs dur
    plot_soft_vs_hard_decision()
    
    print("\n" + "=" * 60)
    print("Toutes les visualisations sont terminées !")
    print("=" * 60)
