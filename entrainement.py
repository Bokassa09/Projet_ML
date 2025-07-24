# entrainement.py
# Données + Modèle + Entraînement 

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

print("🚀 Démarrage de l'entraînement...")

# Créer le dossier pour les résultats
os.makedirs('resultats', exist_ok=True)

# Charger les données MNIST
print("📥 Chargement des données MNIST...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f" Données chargées !")
print(f"   - Images d'entraînement : {x_train.shape}")
print(f"   - Images de test : {x_test.shape}")

# Préparer les données
print("🔧 Préparation des données...")
x_train = x_train.reshape(-1, 28 * 28) / 255.0  # Aplatir et normaliser
x_test = x_test.reshape(-1, 28 * 28) / 255.0

# Convertir les labels en one-hot
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print("Données prêtes !")

# Créer le modèle
print("Création du réseau de neurones...")
modele = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compiler le modèle
modele.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(" Modèle créé !")
modele.summary()

# Entraîner le modèle
print("Début de l'entraînement...")
historique = modele.fit(
    x_train, y_train,
    epochs=1,
    batch_size=32,
    validation_data=(x_test, y_test),
    verbose=1
)

# Sauvegarder le modèle et l'historique
print("Sauvegarde...")
modele.save('resultats/modele.h5')
np.save('resultats/historique.npy', historique.history)

print("Entraînement terminé !")
print("Fichiers sauvegardés dans le dossier 'resultats/'")