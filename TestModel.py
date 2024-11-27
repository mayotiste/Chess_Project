import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

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
model = load_model(r"C:\Users\Utilisateur\OneDrive\Documents\Chess\chessmodel.keras")  # Mettez le chemin correct du modèle
label_encoder = LabelEncoder()
label_encoder.fit(["Dxe5","e1","e2","e3","e6","e4=D","h=5#", "e4", "Fxe4", "Fxe6", "o-o", "Rfx4", "RxF4"])  # Exemple d'étiquettes

# Répertoire des images
images_dir = r"C:\Users\Utilisateur\OneDrive\Documents\Chess\image_concatenee\images"

# Obtenez toutes les images du répertoire
image_paths = os.listdir(images_dir)

# Sélectionner 50 images au hasard
random_images = random.sample(image_paths, 50)

# Créer la figure pour afficher les images
fig, axes = plt.subplots(5, 10, figsize=(20, 10))  # Ajustez la taille de la figure en fonction de votre nombre d'images

# Parcourir les images et afficher les prédictions avec les probabilités
for i, (image_name, ax) in enumerate(zip(random_images, axes.flatten())):
    image_path = os.path.join(images_dir, image_name)
    predicted_move, confidence = predict_move(image_path, model, label_encoder)
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap='gray')
    
    # Afficher la prédiction et la probabilité de confiance
    ax.set_title(f"Prediction: {predicted_move}\nConfidence: {confidence*100:.2f}%")
    
    ax.axis('off')

plt.tight_layout()
plt.show()

