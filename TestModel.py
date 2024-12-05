import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from CNN import load_images_and_labels, load_annotations
print("GPUs disponibles :", tf.config.list_physical_devices('GPU'))

# Fonction de prédiction
def predict_move(image_path, model, label_encoder):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 32))  # Redimensionner si nécessaire
    img_normalized = img_resized / 255.0  # Normaliser l'image
    img_input = np.expand_dims(img_normalized, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)
    
    # Effectuer la prédiction
    prediction = model.predict(img_input)
    predicted_index = np.argmax(prediction)  # Index de la classe prédite
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = np.max(prediction)  # Probabilité de la classe prédite
    
    return predicted_label, confidence

# Charger le modèle et le LabelEncoder
model = load_model(r"C:\Users\Utilisateur\OneDrive\Documents\Chess\chessmodel.keras")  
label_encoder = LabelEncoder()
label_encoder.fit(["Ce6", "Cg4+", "Dg1", "Dxa4", "e4", "exb4", "ex3", "Fa2", "Fxg6", "Rg6", "Rxb2", "Ta4", "Txg3"])  

# Répertoire des images
images_dir = r"C:\Users\Utilisateur\OneDrive\Documents\Chess\image_concatenee\images"

# Obtenez toutes les images du répertoire
image_paths = os.listdir(images_dir)

# Sélectionner 50 images au hasard
random_images = random.sample(image_paths, 36)

# Créer la figure pour afficher les images
fig, axes = plt.subplots(6, 6, figsize=(20, 20))  # 6x6 grille pour 36 images

# Variables pour stocker les prédictions et les labels réels
predictions = []
true_labels = []

# Parcourir les images et afficher les prédictions avec les probabilités
for i, (image_name, ax) in enumerate(zip(random_images, axes.flatten())):
    image_path = os.path.join(images_dir, image_name)
    predicted_move, confidence = predict_move(image_path, model, label_encoder)
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap='gray')
    
    # Ajouter la prédiction et la probabilité de confiance
    ax.set_title(f"Prediction: {predicted_move}\nConfidence: {confidence*100:.2f}%")
    annotations_file = r'C:\Users\Utilisateur\OneDrive\Documents\Chess\rapport_global.txt'
    ax.axis('off')
    annotations = load_annotations(annotations_file)
    images, labels = load_images_and_labels(images_dir, annotations)
    # Ajouter les prédictions et les labels réels aux listes pour l'évaluation
    true_labels.append(labels)  # Supposons que le nom de l'image commence par le label
    predictions.append(predicted_move)

plt.tight_layout()
plt.show()

# Calcul des métriques de performance
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted', labels=np.unique(predictions))
recall = recall_score(true_labels, predictions, average='weighted', labels=np.unique(predictions))
f1 = f1_score(true_labels, predictions, average='weighted', labels=np.unique(predictions))

# Affichage des résultats
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
