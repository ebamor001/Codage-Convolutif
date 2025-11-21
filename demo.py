"""
Démonstration rapide du système de codes convolutifs
"""
import numpy as np
from convolutional_encoder import ConvolutionalEncoder
from viterbi_decoder import ViterbiDecoder

print("=" * 60)
print("DÉMONSTRATION - Code Convolutif avec Décodeur de Viterbi")
print("=" * 60)

# Créer l'encodeur (K=3, rate 1/2 - standard NASA)
encoder = ConvolutionalEncoder(constraint_length=3, generator_polynomials=[7, 5])
decoder = ViterbiDecoder(encoder)

print("\n1. Configuration de l'encodeur:")
print(f"   Longueur de contrainte: K = {encoder.K}")
print(f"   Polynômes générateurs: {encoder.generators} (octal)")
print(f"   Taux de codage: 1/{encoder.n}")
print(f"   Nombre d'états: {encoder.get_num_states()}")

# Message simple
message = "HELLO"
print(f"\n2. Message à transmettre: '{message}'")

# Convertir en bits
bits = np.unpackbits(np.frombuffer(message.encode(), dtype=np.uint8))
print(f"   Longueur en bits: {len(bits)} bits")

# Encoder
encoded = encoder.encode(bits)
print(f"\n3. Après encodage: {len(encoded)} bits (taux 1/{encoder.n})")

# Simuler quelques erreurs
corrupted = encoded.copy()
error_positions = [10, 25, 40]
for pos in error_positions:
    if pos < len(corrupted):
        corrupted[pos] = 1 - corrupted[pos]
print(f"\n4. Simulation: {len(error_positions)} erreurs introduites")

# Décoder
decoded = decoder.decode(corrupted)
print(f"\n5. Après décodage: {len(decoded)} bits")

# Reconstruire le message
decoded_bytes = np.packbits(decoded[:len(bits)])
try:
    decoded_message = decoded_bytes.tobytes().decode('utf-8', errors='ignore')
    print(f"   Message décodé: '{decoded_message}'")
except:
    print(f"   Message décodé: (erreurs restantes)")

# Vérifier
errors = np.sum(bits != decoded[:len(bits)])
print(f"\n6. Résultat: {errors} erreurs après décodage")
if errors == 0:
    print("   ✓ Transmission parfaite malgré le bruit!")
else:
    print(f"   ⚠ Quelques erreurs non corrigées ({errors}/{len(bits)})")

print("\n" + "=" * 60)
print("Décodage réussi avec l'algorithme de Viterbi!")
print("=" * 60)
