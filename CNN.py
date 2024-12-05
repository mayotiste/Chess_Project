import os
import cv2
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Chemins des fichiers
images_dir = r"C:\Users\Utilisateur\OneDrive\Documents\Chess\image_concatenee\images"
annotations_file = r"C:\Users\Utilisateur\OneDrive\Documents\Chess\rapport_global.txt"

# Fonction pour charger les annotations à partir du fichier
def load_annotations(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            if not line.startswith("#"):  # Ignorer les lignes de commentaires
                # Utiliser une expression régulière pour extraire les informations
                match = re.match(r"(\S+) \S+ \d+ \d+ \(([\d, ]+)\) \S+ (\S+)", line)
                if match:
                    image_id = match.group(1)  # Extraire l'ID de l'image
                    bounding_box = tuple(map(int, match.group(2).split(",")))  # Extraire la bounding box
                    label = match.group(3)  # Extraire la transcription (label)
                    data.append((image_id, label, bounding_box))  # Ajouter les informations extraites à la liste
                else:
                    print(f"Ligne mal formée : {line.strip()}")  # Afficher un message d'erreur si la ligne ne correspond pas
    # Retourner un DataFrame contenant les informations extraites
    return pd.DataFrame(data, columns=["image_id", "label", "bounding_box"])

# Charger les annotations à partir du fichier
annotations = load_annotations(annotations_file)
print(annotations)

# Fonction pour charger les images et les labels associés
def load_images_and_labels(images_dir, annotations):
    images = []
    labels = []
    # Parcourir chaque ligne du DataFrame des annotations
    for _, row in annotations.iterrows():
        # Construire le chemin complet de l'image à partir de l'ID de l'image
        image_path = os.path.join(images_dir, row["image_id"] + ".png")
        # Vérifier si l'image existe
        if os.path.exists(image_path):
            # Charger l'image en niveaux de gris
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)  # Ajouter l'image à la liste des images
            labels.append(row["label"])  # Ajouter le label correspondant à la liste des labels
    # Retourner un tableau numpy des images et des labels
    return np.array(images), labels

# Charger les images et les labels
images, labels = load_images_and_labels(images_dir, annotations)

# Normaliser les images (valeurs entre 0 et 1)
images = images / 255.0  

# Ajouter une dimension supplémentaire pour les canaux (image en niveaux de gris)
images = np.expand_dims(images, axis=-1)

# Encodage des labels sous forme numérique (étiquettes catégorielles)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)  # Encoder les labels sous forme numérique
categorical_labels = to_categorical(encoded_labels)  # Convertir les labels en format one-hot

# Diviser les données en ensembles d'entraînement et de test (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(images, categorical_labels, test_size=0.2, random_state=42)


# Création du modèle
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model((32, 128, 1), len(label_encoder.classes_))
model.summary()


# Exemple de prédiction
def predict_move(image_path, model, label_encoder):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 32)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=(0, -1))
    prediction = model.predict(img_resized)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_label

example_image = os.path.join(images_dir, "a01-000u-00-102.png")
predicted_move = predict_move(example_image, model, label_encoder)
print(f"Predicted move: {predicted_move}")

def main():
    # Charger les annotations à partir du fichier
    annotations = load_annotations(annotations_file)
    print(annotations)

    # Charger les images et les labels
    images, labels = load_images_and_labels(images_dir, annotations)

    # Normaliser les images (valeurs entre 0 et 1)
    images = images / 255.0  

    # Ajouter une dimension supplémentaire pour les canaux (image en niveaux de gris)
    images = np.expand_dims(images, axis=-1)

    # Encodage des labels sous forme numérique (étiquettes catégorielles)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)  # Encoder les labels sous forme numérique
    categorical_labels = to_categorical(encoded_labels)  # Convertir les labels en format one-hot

    # Diviser les données en ensembles d'entraînement et de test (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(images, categorical_labels, test_size=0.2, random_state=42)

    # Création du modèle
    model = create_model((32, 128, 1), len(label_encoder.classes_))
    model.summary()

    # Entraînement du modèle
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=32)

    # Évaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # Sauvegarder le modèle
    model.save("chessmodel.keras")

    # Exemple de prédiction
    example_image = os.path.join(images_dir, "a01-000u-00-102.png")
    predicted_move = predict_move(example_image, model, label_encoder)
    print(f"Predicted move: {predicted_move}")

# Vérifier si le fichier est exécuté directement (pas importé)
if __name__ == "__main__":
    main()
