# graphiques.py
# Visualisations sympas du modèle MNIST

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.metrics import ConfusionMatrixDisplay
import random

print(" Création des graphiques...")

# Vérifier que les fichiers existent
try:
    historique = np.load('resultats/historique.npy', allow_pickle=True).item()
    matrice_confusion = np.load('resultats/matrice_confusion.npy')
    predictions = np.load('resultats/predictions.npy')
    vraies_classes = np.load('resultats/vraies_classes.npy')
    print(" Données chargées !")
except:
    print(" Erreur : Lancez d'abord 'entrainement.py' et 'test.py' !")
    exit()

# Charger les images originales pour l'affichage
print(" Chargement des images...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Configuration des graphiques
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

# 1. COURBES D'ENTRAÎNEMENT
print(" Graphique 1 : Courbes d'entraînement...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Précision
ax1.plot(historique['accuracy'], 'b-', label='Entraînement', linewidth=2)
ax1.plot(historique['val_accuracy'], 'r-', label='Validation', linewidth=2)
ax1.set_title('Évolution de la Précision', fontsize=16, fontweight='bold')
ax1.set_xlabel('Époque')
ax1.set_ylabel('Précision')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Perte
ax2.plot(historique['loss'], 'b-', label='Entraînement', linewidth=2)
ax2.plot(historique['val_loss'], 'r-', label='Validation', linewidth=2)
ax2.set_title('Évolution de la Perte', fontsize=16, fontweight='bold')
ax2.set_xlabel('Époque')
ax2.set_ylabel('Perte')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultats/courbes_entrainement.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. MATRICE DE CONFUSION
print(" Graphique 2 : Matrice de confusion...")
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=matrice_confusion, 
                             display_labels=range(10))
disp.plot(cmap='Blues', values_format='d')
plt.title('Matrice de Confusion', fontsize=16, fontweight='bold')
plt.savefig('resultats/matrice_confusion.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. EXEMPLES D'IMAGES BIEN CLASSÉES
print(" Graphique 3 : Exemples de bonnes prédictions...")
bonnes_predictions = []
for i in range(len(vraies_classes)):
    if vraies_classes[i] == predictions[i]:
        bonnes_predictions.append(i)

# Prendre 12 exemples au hasard
echantillon_bons = random.sample(bonnes_predictions, 12)

plt.figure(figsize=(12, 8))
for i, idx in enumerate(echantillon_bons):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f'Prédit: {predictions[idx]} ', color='green', fontweight='bold')
    plt.axis('off')

plt.suptitle('Exemples de Bonnes Prédictions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('resultats/bonnes_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. EXEMPLES D'ERREURS
print(" Graphique 4 : Exemples d'erreurs...")
erreurs = []
for i in range(len(vraies_classes)):
    if vraies_classes[i] != predictions[i]:
        erreurs.append(i)

if len(erreurs) > 0:
    # Prendre les 12 premières erreurs
    echantillon_erreurs = erreurs[:12]
    
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(echantillon_erreurs):
        plt.subplot(3, 4, i + 1)
        plt.imshow(x_test[idx], cmap='gray')
        plt.title(f'Vrai: {vraies_classes[idx]} → Prédit: {predictions[idx]}', 
                 color='red', fontweight='bold')
        plt.axis('off')
    
    plt.suptitle('Exemples d\'Erreurs de Classification', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('resultats/erreurs_classification.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print(" Aucune erreur trouvée ! Modèle parfait !")

# 5. STATISTIQUES PAR CHIFFRE
print(" Graphique 5 : Performance par chiffre...")
precision_par_chiffre = []
for chiffre in range(10):
    indices_chiffre = np.where(vraies_classes == chiffre)[0]
    if len(indices_chiffre) > 0:
        bonnes_pred = np.sum(predictions[indices_chiffre] == chiffre)
        precision = bonnes_pred / len(indices_chiffre)
        precision_par_chiffre.append(precision)
    else:
        precision_par_chiffre.append(0)

plt.figure(figsize=(10, 6))
bars = plt.bar(range(10), precision_par_chiffre, color='skyblue', edgecolor='navy')
plt.title('Précision par Chiffre', fontsize=16, fontweight='bold')
plt.xlabel('Chiffre')
plt.ylabel('Précision')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Ajouter les valeurs sur les barres
for i, (bar, precision) in enumerate(zip(bars, precision_par_chiffre)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{precision:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('resultats/precision_par_chiffre.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. ÉCHANTILLONS ALÉATOIRES DU DATASET
print(" Graphique 6 : Échantillon du dataset...")
indices_aleatoires = random.sample(range(len(x_test)), 15)

plt.figure(figsize=(12, 9))
for i, idx in enumerate(indices_aleatoires):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    couleur = 'green' if vraies_classes[idx] == predictions[idx] else 'red'
    plt.title(f'Vrai: {vraies_classes[idx]} | Prédit: {predictions[idx]}', 
             color=couleur, fontsize=10)
    plt.axis('off')

plt.suptitle('Échantillon Aléatoire du Dataset de Test', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('resultats/echantillon_dataset.png', dpi=300, bbox_inches='tight')
plt.show()

print(" Tous les graphiques ont été créés !")
print("\n Images sauvegardées :")
print("   • courbes_entrainement.png - Évolution pendant l'entraînement")
print("   • matrice_confusion.png - Matrice de confusion")
print("   • bonnes_predictions.png - Exemples de bonnes prédictions")
print("   • erreurs_classification.png - Exemples d'erreurs")
print("   • precision_par_chiffre.png - Performance par chiffre")
print("   • echantillon_dataset.png - Échantillon aléatoire")

print("\n Visualisations terminées !")
print(" Regardez le dossier 'resultats/' pour voir tous vos fichiers !")