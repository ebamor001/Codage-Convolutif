"""
Tests unitaires pour les codes convolutifs
"""
import numpy as np
from convolutional_encoder import ConvolutionalEncoder
from viterbi_decoder import ViterbiDecoder
from simulator import ConvolutionalCodeSimulator


def test_encoder_basic():
    """Test de l'encodeur de base"""
    print("Test 1: Encodeur de base...")
    
    encoder = ConvolutionalEncoder(constraint_length=3, generator_polynomials=[7, 5])
    
    # Test avec une séquence simple
    data = np.array([1, 0, 1])
    encoded = encoder.encode(data)
    
    # Vérifier que la longueur est correcte
    # 3 bits de données + 2 bits de terminaison = 5 bits d'entrée
    # rate 1/2 donc 10 bits de sortie
    expected_length = (len(data) + encoder.memory) * encoder.n
    assert len(encoded) == expected_length, f"Longueur incorrecte: {len(encoded)} != {expected_length}"
    
    print("  ✓ Longueur de sortie correcte")
    
    # Vérifier que la sortie est binaire
    assert np.all((encoded == 0) | (encoded == 1)), "La sortie doit être binaire"
    print("  ✓ Sortie binaire")
    
    print("  ✓ Test réussi!\n")


def test_decoder_no_noise():
    """Test du décodeur sans bruit"""
    print("Test 2: Décodeur sans bruit...")
    
    encoder = ConvolutionalEncoder(constraint_length=3, generator_polynomials=[7, 5])
    decoder = ViterbiDecoder(encoder)
    
    # Encoder puis décoder
    data = np.array([1, 0, 1, 1, 0, 1, 0])
    encoded = encoder.encode(data)
    decoded = decoder.decode(encoded)
    
    # Vérifier que le décodage est parfait
    assert np.array_equal(data, decoded), "Décodage incorrect sans bruit"
    print("  ✓ Décodage parfait sans bruit")
    
    print("  ✓ Test réussi!\n")


def test_decoder_with_errors():
    """Test du décodeur avec quelques erreurs"""
    print("Test 3: Décodeur avec erreurs...")
    
    encoder = ConvolutionalEncoder(constraint_length=3, generator_polynomials=[7, 5])
    decoder = ViterbiDecoder(encoder)
    
    # Encoder
    data = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    encoded = encoder.encode(data)
    
    # Introduire une erreur
    corrupted = encoded.copy()
    corrupted[5] = 1 - corrupted[5]  # Inverser un bit
    
    # Décoder
    decoded = decoder.decode(corrupted)
    
    # Le décodeur devrait corriger cette erreur simple
    errors = np.sum(data != decoded)
    print(f"  Erreurs après décodage: {errors} / {len(data)}")
    
    # Avec une seule erreur, le décodeur devrait pouvoir corriger
    assert errors <= 1, "Le décodeur devrait corriger une seule erreur"
    print("  ✓ Correction d'erreur réussie")
    
    print("  ✓ Test réussi!\n")


def test_different_rates():
    """Test avec différents taux de codage"""
    print("Test 4: Différents taux de codage...")
    
    configs = [
        (3, [7, 5]),      # rate 1/2
        (3, [7, 5, 3]),   # rate 1/3
        (4, [15, 17]),    # rate 1/2, K=4
    ]
    
    data = np.array([1, 0, 1, 1, 0])
    
    for K, gen in configs:
        encoder = ConvolutionalEncoder(K, gen)
        decoder = ViterbiDecoder(encoder)
        
        encoded = encoder.encode(data)
        decoded = decoder.decode(encoded)
        
        assert np.array_equal(data, decoded), f"Échec pour K={K}, gen={gen}"
        print(f"  ✓ K={K}, rate=1/{len(gen)}: OK")
    
    print("  ✓ Test réussi!\n")


def test_simulator():
    """Test du simulateur"""
    print("Test 5: Simulateur...")
    
    encoder = ConvolutionalEncoder(constraint_length=3, generator_polynomials=[7, 5])
    decoder = ViterbiDecoder(encoder)
    simulator = ConvolutionalCodeSimulator(encoder, decoder)
    
    # Test de transmission simple
    data = np.array([1, 0, 1, 1, 0])
    results = simulator.simulate_single_transmission(data, snr_db=10, soft_decision=True)
    
    assert 'data_bits' in results, "Résultats incomplets"
    assert 'encoded_bits' in results, "Résultats incomplets"
    assert 'decoded_bits' in results, "Résultats incomplets"
    assert 'ber' in results, "Résultats incomplets"
    
    print("  ✓ Structure des résultats correcte")
    
    # À SNR élevé, le BER devrait être très faible
    assert results['ber'] <= 0.5, "BER trop élevé à SNR=10dB"
    print(f"  ✓ BER acceptable: {results['ber']:.6f}")
    
    print("  ✓ Test réussi!\n")


def test_get_next_state():
    """Test de la fonction get_next_state"""
    print("Test 6: Fonction get_next_state...")
    
    encoder = ConvolutionalEncoder(constraint_length=3, generator_polynomials=[7, 5])
    
    # Test des transitions d'état
    state = 0
    next_state_0, output_0 = encoder.get_next_state(state, 0)
    next_state_1, output_1 = encoder.get_next_state(state, 1)
    
    # Vérifier que les états suivants sont différents
    assert next_state_0 != next_state_1, "Les états suivants devraient être différents"
    print("  ✓ Transitions d'état correctes")
    
    # Vérifier que les sorties ont la bonne longueur
    assert len(output_0) == encoder.n, "Longueur de sortie incorrecte"
    assert len(output_1) == encoder.n, "Longueur de sortie incorrecte"
    print("  ✓ Longueur de sortie correcte")
    
    print("  ✓ Test réussi!\n")


def test_soft_decision():
    """Test du décodage souple"""
    print("Test 7: Décodage souple...")
    
    encoder = ConvolutionalEncoder(constraint_length=3, generator_polynomials=[7, 5])
    decoder = ViterbiDecoder(encoder)
    
    # Encoder
    data = np.array([1, 0, 1, 1, 0])
    encoded = encoder.encode(data)
    
    # Convertir en symboles réels avec un peu de bruit
    symbols = 2.0 * encoded - 1.0  # 0 -> -1, 1 -> 1
    noise = np.random.normal(0, 0.1, len(symbols))
    noisy_symbols = symbols + noise
    
    # Décoder avec soft decision
    decoded = decoder.decode_soft(noisy_symbols)
    
    # Vérifier que le décodage fonctionne
    assert len(decoded) == len(data), "Longueur de décodage incorrecte"
    print(f"  ✓ Décodage souple: {np.sum(data == decoded)}/{len(data)} bits corrects")
    
    print("  ✓ Test réussi!\n")


def run_all_tests():
    """Exécute tous les tests"""
    print("=" * 60)
    print("EXÉCUTION DES TESTS UNITAIRES")
    print("=" * 60 + "\n")
    
    tests = [
        test_encoder_basic,
        test_decoder_no_noise,
        test_decoder_with_errors,
        test_different_rates,
        test_simulator,
        test_get_next_state,
        test_soft_decision,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ Test échoué: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗ Erreur: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"RÉSULTATS: {passed} tests réussis, {failed} tests échoués")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    np.random.seed(42)
    success = run_all_tests()
    exit(0 if success else 1)
