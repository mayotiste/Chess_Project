import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# 1. Préparer les données

# Répertoires des images et fichier .txt
txt_file = r'C:\Users\Utilisateur\OneDrive\Documents\Chess\rapport_global.txt'
image_dir = r'C:\Users\Utilisateur\OneDrive\Documents\Chess\image_concatenee\images'

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

# 2. Identifier les classes (transcriptions distinctes)
transcriptions = df['transcription'].unique()  # Obtenez toutes les transcriptions uniques
transcription_to_label = {transcription: i for i, transcription in enumerate(transcriptions)}  # Mappage transcription -> label
label_to_transcription = {i: transcription for transcription, i in transcription_to_label.items()}  # Mappage inverse

# 3. Prétraiter les images et préparer les étiquettes
def preprocess_image(image_id, image_dir, target_size=(128, 32)):
    """Charger et redimensionner l'image."""
    image_path = os.path.join(image_dir, f"{image_id}.png")
    if not os.path.exists(image_path):
        return None
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Lire en niveaux de gris
    if image is None:
        return None  # Retourner None si l'image ne peut pas être lue
    
    image = cv2.resize(image, target_size)  # Redimensionner
    image = np.expand_dims(image, axis=-1)  # Ajouter un canal (niveaux de gris)
    image = image / 255.0  # Normaliser les pixels
    return image

# Encoder les transcriptions
characters = "-#+=12345678abCDefghoRTx"  # Par exemple, lettres et chiffres pour les coups d'échecs
char_to_index = {char: i for i, char in enumerate(characters)}
index_to_char = {i: char for i, char in enumerate(characters)}

def encode_transcription(transcription):
    return [char_to_index[char] for char in transcription]

# Exemple de préparation des données pour l'entraînement
X_data = []
y_data = []
y_lengths = []  # Longueurs des transcriptions
labels_data = []  # Labels des catégories (transcriptions)

for _, row in df.iterrows():
    image_id = row["image_id"]
    transcription = row["transcription"]
    
    # Charger l'image et la prétraiter
    image = preprocess_image(image_id, image_dir)
    
    if image is None:
        continue
    
    X_data.append(image)
    
    # Encoder la transcription et calculer la longueur
    encoded_transcription = encode_transcription(transcription)
    y_data.append(encoded_transcription)
    y_lengths.append(len(encoded_transcription))
    
    # Ajouter l'étiquette de classe pour chaque transcription
    label = transcription_to_label[transcription]
    labels_data.append(label)

# Convertir les listes en tableaux numpy
X_data = np.array(X_data)
y_data = np.array(y_data)
y_lengths = np.array(y_lengths)
labels_data = np.array(labels_data)  # Ajout de l'étiquette de la transcription

# 4. Séparer les données en ensembles d'entraînement, validation et test
X_train, X_temp, y_train, y_temp, labels_train, labels_temp = train_test_split(X_data, y_data, labels_data, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test, labels_val, labels_test = train_test_split(X_temp, y_temp, labels_temp, test_size=0.5, random_state=42)

# 5. Appliquer le padding sur les séquences de transcriptions
# Appliquer le padding sur les séquences de transcriptions
# Appliquer le padding sur les séquences de transcriptions
max_transcription_length = max(y_lengths)  # Trouver la longueur maximale des transcriptions

def pad_transcription(transcription, max_length):
    """Compléter la transcription avec des 0 jusqu'à ce qu'elle atteigne la longueur maximale"""
    # Assurez-vous que la transcription n'est pas vide avant de faire le padding
    if len(transcription) == 0:
        print("Transcription vide détectée !")
        return [0] * max_length  # Remplir avec des 0 si la transcription est vide
    return transcription + [0] * (max_length - len(transcription))

# Filtrer et appliquer le padding aux données
def filter_and_pad(y_data, max_length):
    """Filtrer les données vides et appliquer le padding"""
    return np.array([pad_transcription(transcription, max_length) for transcription in y_data if len(transcription) > 0])

# Filtrer et appliquer le padding sur les ensembles de données
y_train_padded = filter_and_pad(y_train, max_transcription_length)
y_val_padded = filter_and_pad(y_val, max_transcription_length)
y_test_padded = filter_and_pad(y_test, max_transcription_length)

# Vérification des formes après padding
print(f"Forme de y_train_padded: {y_train_padded.shape}")
print(f"Forme de y_val_padded: {y_val_padded.shape}")
print(f"Forme de y_test_padded: {y_test_padded.shape}")



# 6. Créer le modèle BiLSTM + CTC avec TensorFlow
def create_ctc_model(input_shape, num_classes):
    input_layer = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Reshape(target_shape=(-1, 64))(x)
    
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=input_layer, outputs=x)
    
    # Fonction de perte CTC avec TensorFlow
    def ctc_loss(y_true, y_pred):
        # Calcul des longueurs de chaque transcription
        y_lengths = tf.cast(tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.float32), axis=-1), tf.int32)
        
        # Remplacer np.ones_like par tf.ones_like
        ones = tf.ones_like(y_lengths, dtype=tf.int32) * X_train.shape[1]  # Crée un tensor de 1s avec la même forme que y_lengths
        
        # Calcul de la perte CTC
        return tf.reduce_mean(K.ctc_batch_cost(y_true, y_pred, y_lengths, ones))
    
    model.compile(optimizer='adam', loss=ctc_loss)
    
    return model

# Créer le modèle
input_shape = (32, 128, 1)
num_classes = len(characters)  # Nombre de classes (caractères)
model = create_ctc_model(input_shape, num_classes)

# Afficher le résumé du modèle
model.summary()

# 7. Entraîner le modèle
print("Entraînement du modèle...")
history = model.fit(X_train, y_train_padded, validation_data=(X_val, y_val_padded), batch_size=16, epochs=10)

# 8. Évaluer le modèle
print("Évaluation du modèle...")
test_loss = model.evaluate(X_test, y_test_padded)
print(f"Test Loss: {test_loss}")
