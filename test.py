# test.py
# Test du modèle + Sauvegarde des résultats

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report

print(" Test du modèle entraîné...")

# Vérifier que le modèle existe
try:
    modele = tf.keras.models.load_model('resultats/modele.h5')
    print(" Modèle chargé avec succès !")
except:
    print(" Erreur : Lancez d'abord 'entrainement.py' !")
    exit()

# Charger les données de test
print("Chargement des données de test...")
(_, _), (x_test, y_test) = mnist.load_data()

# Préparer les données (même préparation que l'entraînement)
x_test_prep = x_test.reshape(-1, 28 * 28) / 255.0
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

print("Données de test prêtes !")

# Tester le modèle
print(" Évaluation du modèle...")
perte, precision = modele.evaluate(x_test_prep, y_test_onehot, verbose=0)

print(f"\n RÉSULTATS :")
print(f"   • Précision : {precision:.4f} ({precision*100:.2f}%)")
print(f"   • Perte : {perte:.4f}")

# Faire des prédictions
print(" Génération des prédictions...")
predictions = modele.predict(x_test_prep, verbose=0)
y_pred = np.argmax(predictions, axis=1)

# Calculer la matrice de confusion
matrice_confusion = confusion_matrix(y_test, y_pred)

# Rapport de classification
rapport = classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)])

# Trouver quelques erreurs intéressantes
erreurs = []
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        erreurs.append({
            'index': i,
            'vraie_classe': y_test[i],
            'prediction': y_pred[i],
            'confiance': np.max(predictions[i])
        })

print(f" Nombre d'erreurs : {len(erreurs)} sur {len(y_test)}")

# Sauvegarder tous les résultats
print(" Sauvegarde des résultats...")

# Sauvegarder les métriques
with open('resultats/metriques.txt', 'w') as f:
    f.write(f"RÉSULTATS DU TEST\n")
    f.write(f"=================\n\n")
    f.write(f"Précision : {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Perte : {perte:.4f}\n")
    f.write(f"Erreurs : {len(erreurs)}/{len(y_test)}\n\n")
    f.write("RAPPORT DÉTAILLÉ\n")
    f.write("================\n")
    f.write(rapport)

# Sauvegarder les prédictions et erreurs
np.save('resultats/predictions.npy', y_pred)
np.save('resultats/vraies_classes.npy', y_test)
np.save('resultats/matrice_confusion.npy', matrice_confusion)

# Sauvegarder quelques erreurs intéressantes
erreurs_sample = erreurs[:20]  # Garder les 20 premières erreurs
with open('resultats/erreurs_exemples.txt', 'w') as f:
    f.write("EXEMPLES D'ERREURS\n")
    f.write("==================\n\n")
    for err in erreurs_sample:
        f.write(f"Image {err['index']} : Vraie classe = {err['vraie_classe']}, "
                f"Prédiction = {err['prediction']}, Confiance = {err['confiance']:.3f}\n")

print(" Résultats sauvegardés !")
print("\n Fichiers créés :")
print("   • metriques.txt - Résultats principaux")
print("   • predictions.npy - Toutes les prédictions")
print("   • matrice_confusion.npy - Matrice de confusion")
print("   • erreurs_exemples.txt - Exemples d'erreurs")

print(f"\n Test terminé ! Précision : {precision*100:.2f}%")