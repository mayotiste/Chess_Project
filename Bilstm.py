import os
import cv2
import pandas as pd
import numpy as np
from keras import layers, models, backend as K

# 1. Préparer les données

# Répertoires des images et fichier .txt
txt_file = r'C:\Users\chess2425\Downloads\Chess_Project-Mael\Chess_Project-Mael\rapport_global.txt'
image_dir = r'C:\Users\chess2425\Downloads\Chess_Project-Mael\Chess_Project-Mael\image_concatenee\images'

# Lire le fichier txt
data = []
with open(txt_file, 'r') as file:
    for line in file.readlines():
        if not line.startswith('#'):  # Ignorer les commentaires
            parts = line.split()
            image_id = parts[0]  # a01-000u-00-301
            transcription = parts[-1]  # DR54
            bbox = parts[4]  # (11, 0, 120, 31)
            bbox = tuple(map(int, bbox[1:-1].split(',')))  # Convertir en tuple (x, y, w, h)
            data.append((image_id, transcription, bbox))

# Convertir les données en DataFrame
df = pd.DataFrame(data, columns=["image_id", "transcription", "bbox"])

# Afficher un exemple pour vérifier
print(df.head())

# 2. Prétraiter les images et préparer les étiquettes
def preprocess_image(image_id, image_dir, target_size=(128, 32)):
    """Charger et redimensionner l'image."""
    image_path = os.path.join(image_dir, f"{image_id}.png")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Lire en niveaux de gris
    image = cv2.resize(image, target_size)  # Redimensionner
    image = np.expand_dims(image, axis=-1)  # Ajouter un canal (niveaux de gris)
    image = image / 255.0  # Normaliser les pixels
    return image

# 3. Encoder les transcriptions
characters = "-#+=12345678abCdeFghoRTx"  # Par exemple, lettres et chiffres pour les coups d'échecs
char_to_index = {char: i for i, char in enumerate(characters)}
index_to_char = {i: char for i, char in enumerate(characters)}

# Encoder les transcriptions en indices
def encode_transcription(transcription):
    return [char_to_index[char] for char in transcription]

# Exemple de préparation des données pour l'entraînement
X_train = []
y_train = []
y_lengths = []  # Longueurs des transcriptions

for _, row in df.iterrows():
    image_id = row["image_id"]
    transcription = row["transcription"]
    
    # Charger l'image et la prétraiter
    image = preprocess_image(image_id, image_dir)
    X_train.append(image)
    
    # Encoder la transcription et calculer la longueur
    encoded_transcription = encode_transcription(transcription)
    y_train.append(encoded_transcription)
    y_lengths.append(len(encoded_transcription))

# Convertir les listes en tableaux numpy
X_train = np.array(X_train)
y_train = np.array(y_train)
y_lengths = np.array(y_lengths)

# 4. Définir le modèle BiLSTM + CTC

def create_ctc_model(input_shape, num_classes):
    """Créer le modèle BiLSTM + CTC."""
    input_layer = layers.Input(shape=input_shape)
    
    # Couches CNN pour extraire des caractéristiques
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Applatissement pour les LSTM
    x = layers.Reshape(target_shape=(-1, 64))(x)
    
    # LSTM bidirectionnels
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    
    # Couche dense pour la classification des caractères
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    # Créer le modèle
    model = models.Model(inputs=input_layer, outputs=x)
    
    # Fonction de perte CTC
    def ctc_loss(y_true, y_pred):
        return K.mean(K.ctc_batch_cost(y_true, y_pred, y_lengths, np.ones_like(y_lengths) * X_train.shape[1]))
    
    # Compiler le modèle
    model.compile(optimizer='adam', loss=ctc_loss)
    
    return model

# Créer le modèle
input_shape = (32, 128, 1)  # Image redimensionnée (hauteur, largeur, canal)
num_classes = len(characters)  # Nombre de classes (caractères possibles)
model = create_ctc_model(input_shape, num_classes)

# Afficher le résumé du modèle
model.summary()

# 5. Entraîner le modèle
model.fit(X_train, y_train, batch_size=16, epochs=10)
