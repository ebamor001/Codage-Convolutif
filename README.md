# Codage Convolutif avec Décodeur de Viterbi

Implémentation en Python de codes convolutifs avec décodage de Viterbi pour la correction d'erreurs dans les communications numériques.

## Description

Ce projet fournit une implémentation complète de :
- **Encodeur convolutif** : Encode les données binaires avec des codes convolutifs configurables
- **Décodeur de Viterbi** : Décode les données reçues avec l'algorithme optimal de Viterbi
- **Simulateur** : Simule des transmissions sur canal bruité (AWGN) et mesure les performances
- **Exemples et visualisations** : Scripts pour tester et visualiser les performances

## Caractéristiques

- ✅ Support de différentes longueurs de contrainte (K)
- ✅ Support de différents taux de codage (rate 1/2, 1/3, etc.)
- ✅ Décodage souple (soft decision) et dur (hard decision)
- ✅ Simulation avec bruit AWGN (Additive White Gaussian Noise)
- ✅ Calcul du taux d'erreur binaire (BER)
- ✅ Génération de courbes de performance
- ✅ Polynômes générateurs configurables

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/ebamor001/Codage-Convolutif.git
cd Codage-Convolutif

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Exemple simple

```python
import numpy as np
from convolutional_encoder import ConvolutionalEncoder
from viterbi_decoder import ViterbiDecoder

# Créer un encodeur (K=3, rate 1/2)
encoder = ConvolutionalEncoder(constraint_length=3, generator_polynomials=[7, 5])

# Données à encoder
data = np.array([1, 0, 1, 1, 0])

# Encoder
encoded = encoder.encode(data)

# Créer un décodeur
decoder = ViterbiDecoder(encoder)

# Décoder
decoded = decoder.decode(encoded)

print(f"Original : {data}")
print(f"Décodé   : {decoded}")
```

### Exécuter les exemples

```bash
# Exemples de base
python examples.py

# Générer des courbes de performance
python plot_performance.py
```

### Simulation avec bruit

```python
from simulator import ConvolutionalCodeSimulator

# Créer le simulateur
simulator = ConvolutionalCodeSimulator(encoder, decoder)

# Simuler une transmission avec SNR = 5 dB
results = simulator.simulate_single_transmission(data, snr_db=5, soft_decision=True)

print(f"Erreurs : {results['bit_errors']}")
print(f"BER     : {results['ber']:.6f}")
```

## Structure du code

```
.
├── convolutional_encoder.py   # Encodeur convolutif
├── viterbi_decoder.py         # Décodeur de Viterbi
├── simulator.py               # Simulateur avec canal bruité
├── examples.py                # Exemples d'utilisation
├── plot_performance.py        # Génération de graphiques
├── requirements.txt           # Dépendances Python
└── README.md                  # Ce fichier
```

## Codes convolutifs supportés

### Code NASA standard (K=3, rate 1/2)
```python
encoder = ConvolutionalEncoder(constraint_length=3, generator_polynomials=[7, 5])
```
- Polynômes : (111, 101) en binaire
- Taux : 1/2 (2 bits sortie pour 1 bit entrée)
- États : 4

### Code rate 1/3 (K=3)
```python
encoder = ConvolutionalEncoder(constraint_length=3, generator_polynomials=[7, 5, 3])
```
- Taux : 1/3 (3 bits sortie pour 1 bit entrée)
- Meilleure protection mais débit réduit

### Code K=4 (rate 1/2)
```python
encoder = ConvolutionalEncoder(constraint_length=4, generator_polynomials=[15, 17])
```
- Plus de mémoire, meilleures performances
- États : 8

## Algorithme de Viterbi

Le décodeur utilise l'algorithme de Viterbi qui :
1. Calcule les métriques de chemin pour tous les états possibles
2. Conserve le meilleur chemin (distance de Hamming minimale)
3. Effectue un retour en arrière pour récupérer la séquence décodée

**Décodage souple** : Utilise les valeurs réelles du signal pour de meilleures performances
**Décodage dur** : Utilise des décisions binaires (0/1)

## Performances typiques

Avec le code K=3 rate 1/2 et décodage souple :
- À SNR = 5 dB : BER ≈ 10⁻³
- À SNR = 7 dB : BER ≈ 10⁻⁴
- Gain de codage : ~5-6 dB par rapport au non codé

## Applications

Les codes convolutifs sont utilisés dans :
- 📡 Communications spatiales (NASA, ESA)
- 📱 Réseaux mobiles (GSM, 3G)
- 📺 Télévision numérique (DVB)
- 💿 Stockage de données
- 🛰️ Communications par satellite

## Références

- **Viterbi, A.J.** (1967). "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm"
- **Lin, S., & Costello, D.J.** (2004). "Error Control Coding"
- **Proakis, J.G., & Salehi, M.** (2008). "Digital Communications"

## Auteur

ebamor001

## Licence

Ce projet est fourni à des fins éducatives et de recherche.