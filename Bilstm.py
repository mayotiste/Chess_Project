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

# Lecture des annotations
def load_annotations(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            if not line.startswith("#"):  # Ignorer les commentaires
                # Utiliser une regex pour extraire les parties de la ligne
                match = re.match(r"(\S+) \S+ \d+ \d+ \(([\d, ]+)\) \S+ (\S+)", line)
                if match:
                    image_id = match.group(1)  # ID de l'image
                    bounding_box = tuple(map(int, match.group(2).split(",")))  # Bounding box
                    label = match.group(3)  # Transcription
                    data.append((image_id, label, bounding_box))
                else:
                    print(f"Ligne mal formée : {line.strip()}")
    return pd.DataFrame(data, columns=["image_id", "label", "bounding_box"])


annotations = load_annotations(annotations_file)
print(annotations)
# Chargement des images et préparation des données
def load_images_and_labels(images_dir, annotations):
    images = []
    labels = []
    for _, row in annotations.iterrows():
        image_path = os.path.join(images_dir, row["image_id"] + ".png")
        print(f"Chargement de l'image : {image_path}")
        if os.path.exists(image_path):
            # Chargement de l'image complète en niveaux de gris
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            labels.append(row["label"])
    return np.array(images), labels

images, labels = load_images_and_labels(images_dir, annotations)
images = images / 255.0  # Normalisation
images = np.expand_dims(images, axis=-1)  # Ajouter une dimension pour les canaux (grayscale)

# Encodage des labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

# Division des données
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
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model((32, 128, 1), len(label_encoder.classes_))

# Entraînement du modèle
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

# Évaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Sauvegarder le modèle
model.save("chessmodel.keras")

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

