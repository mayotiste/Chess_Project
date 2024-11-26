import os
import re
import numpy as np
from matchingfiles import *
from PIL import Image, ImageChops, ImageFilter
import itertools
import random
from skimage import morphology, measure
from txt import *
import matplotlib.pyplot as plt
import shutil 
from rapport import extraire_dernier_groupe, concat_reports
from merge import merge_reports

def show_image(image, title="Image"):
    """Affiche une image avec un titre."""
    plt.imshow(image, cmap='gray' if image.mode == 'L' else None)
    plt.title(title)
    plt.axis('off')  # Ne pas afficher les axes
    plt.show()
def open_image(image):
    """Applique une ouverture (érosion suivie d'une dilatation) pour réduire le bruit."""
    # Convertir l'image en niveaux de gris
    gray_image = image.convert("L")
    #show_image(gray_image, "Image en Niveaux de Gris")

    # Convertir l'image en tableau numpy
    image_array = np.array(gray_image)

    # Binariser l'image (ajuster le seuil selon les besoins)
    binary_image = image_array > 128
    #show_image(Image.fromarray(binary_image.astype(np.uint8) * 255), "Image Binarisée")

    # Appliquer l'ouverture (érosion suivie d'une dilatation) avec une structure plus petite
    opened_image = morphology.binary_opening(binary_image, morphology.square(2))  # Réduit la taille de 3 à 2
    #show_image(Image.fromarray(opened_image.astype(np.uint8) * 255), "Image Ouverte (Érosion + Dilatation)")

    # Supprimer les petits objets en dessous d'une certaine taille (e.g., taille < 64 pixels)
    cleaned_image = morphology.remove_small_objects(opened_image, min_size=64)
    #show_image(Image.fromarray(cleaned_image.astype(np.uint8) * 255), "Image Nettoyée")

    # Convertir le résultat en image PIL
    return Image.fromarray(cleaned_image.astype(np.uint8) * 255)


def resize_image(image, target_size=(128, 32)):
    """Redimensionne l'image à une taille fixe donnée en préservant le ratio d'aspect."""
    # Convertir l'image en mode 'RGB' si elle ne l'est pas déjà
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Obtenir les dimensions de l'image d'origine
    original_size = image.size
    original_ratio = original_size[0] / original_size[1]
    target_ratio = target_size[0] / target_size[1]

    # Calculer la taille de redimensionnement en préservant le ratio
    if original_ratio > target_ratio:
        # L'image est plus large que la cible
        new_width = target_size[0]
        new_height = int(target_size[0] / original_ratio)
    else:
        # L'image est plus haute que la cible
        new_width = int(target_size[1] * original_ratio)
        new_height = target_size[1]

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    #show_image(resized_image, "Image Redimensionnée")
    
    # Optionnel : Si tu veux que l'image redimensionnée soit exactement de la taille cible, tu peux la centrer
    final_image = Image.new('RGB', target_size, (255, 255, 255))  # Fond blanc
    final_image.paste(resized_image, ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2))

    return final_image


def trim_image(image):
    """Recadre les espaces blancs ou transparents autour de l'image après traitement."""
    # Appliquer un filtre médian plus agressif pour réduire le bruit
    filtered_image = image.filter(ImageFilter.MedianFilter(size=3))  # Taille de 5 pour un filtrage plus fort

    # Appliquer l'ouverture (combinaison d'érosion et dilatation)
    processed_image = open_image(filtered_image)

    # Convertir l'image filtrée en mode 'L' (niveaux de gris) et obtenir son masque
    bg = Image.new(processed_image.mode, processed_image.size, processed_image.getpixel((0, 0)))  # Pixel de fond (0, 0)
    diff = ImageChops.difference(processed_image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()

    if bbox:
        cropped_image = processed_image.crop(bbox)
        #show_image(cropped_image, "Image Recadrée boundingbox")

        # Afficher les dimensions de l'image recadrée
        #print(f"Dimensions de l'image recadrée : {cropped_image.size}")  # (largeur, hauteur)
        
        return cropped_image

    #print("Aucune bordure trouvée, l'image reste inchangée")
    return processed_image

def get_next_image_id(output_dir):
    """Détermine l'ID de l'image suivant en vérifiant les fichiers existants dans le dossier."""
    existing_files = os.listdir(output_dir)
    ids = []

    for file_name in existing_files:
        # Utiliser une expression régulière pour extraire l'ID
        match = re.search(r'(\d+)\.png$', file_name)
        if match:
            ids.append(int(match.group(1)))

    # Trouver le plus grand ID et l'incrémenter
    next_id = max(ids) + 1 if ids else 1
    return next_id

def concat_images(images, output_dir, next_id, padding=5, target_size=(128, 32)):
    """Concatène les images données après avoir redimensionné toutes les images à la même taille,
       appliqué 'trim_image', et les enregistre en suivant l'ID incrémenté avec un espacement uniforme,
       tout en ajoutant un padding spécifique à la première image, puis redimensionne l'image combinée."""

    # Ouvrir et convertir toutes les images en RGB
    opened_images = [Image.open(img).convert('RGB') for img in images]
    
    # Appliquer 'trim_image' à chaque image après la conversion
    processed_images = [trim_image(img) for img in opened_images]
    
    # Trouver la largeur et la hauteur maximales parmi toutes les images traitées pour définir la taille uniforme
    max_width = max(img.width for img in processed_images)
    max_height = max(img.height for img in processed_images)
    
    # Redimensionner toutes les images à la même taille (max_width x max_height) tout en maintenant l'aspect ratio
    resized_images = []
    for img in processed_images:
        img_resized = img.resize((max_width, max_height), Image.LANCZOS)  # Fallback for older versions of Pillow
        resized_images.append(img_resized)
    
    # Calculer la largeur totale de l'image concaténée (en tenant compte des images redimensionnées et du padding)
    total_width = sum(img.width for img in resized_images) + (len(resized_images) - 1) * padding
    total_height = max(img.height for img in resized_images)
    
    # Créer une nouvelle image blanche avec la largeur et la hauteur totale
    combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # Coller les images redimensionnées une à une avec un padding entre elles
    x_offset = 10  # Appliquer un padding supplémentaire pour la première image
    for idx, img in enumerate(resized_images):
        combined_image.paste(img, (x_offset, 0))  # Coller l'image à la position x_offset
        x_offset += img.width + padding  # Avancer de la largeur de l'image + le padding

    # Redimensionner l'image combinée à la taille cible après la concaténation
    combined_image_resized = resize_image(combined_image, target_size)
    
    # Générer le nom de fichier avec l'ID calculé
    output_file_name = f"a01-000u-00-{next_id:02d}.png"
    output_path = os.path.join(output_dir, output_file_name)
    
    # Sauvegarder l'image combinée redimensionnée
    combined_image_resized.save(output_path)


def automate_image_combination():
    # Récupère les fichiers et les inputs

    matching_files, search_inputs = main()
    
    # Crée toutes les combinaisons possibles des fichiers
    all_combinations = list(itertools.product(*matching_files.values()))
    
    # Sélectionne un échantillon aléatoire de 100 combinaisons ou toutes si moins de 100
    selected_combinations = random.sample(all_combinations, 100) if len(all_combinations) > 100 else all_combinations

    # Dossier de sortie où les images concaténées seront enregistrées
    output_dir = r"C:\Users\Utilisateur\OneDrive\Documents\Chess\image_concatenee"
    os.makedirs(output_dir, exist_ok=True)

    # Crée un dossier 'images' pour stocker les images
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Obtenez l'ID suivant pour l'enregistrement des fichiers (première image)
    next_id = extraire_dernier_groupe(os.path.join(output_dir, 'images'))

    # Liste pour stocker les chemins des images générées
    image_paths = []

    # Parcourt chaque combinaison et concatène les images
    for idx, combination in enumerate(selected_combinations):
        images_to_concat = list(combination)
        # Pour la première image, on utilise next_id
        if idx == 0:
            # Concatène la première image et l'enregistre avec next_id
            next_id += 1
            concat_images(images_to_concat, output_dir, next_id)
            image_paths.append(os.path.join(output_dir, f"a01-000u-00-{next_id:02d}.png"))
        else:
            # Pour les images suivantes, on utilise last_id (ID mis à jour après chaque image)
            last_id = extraire_dernier_groupe(output_dir) + 1 # Dernier ID utilisé dans le dossier
            concat_images(images_to_concat, output_dir, last_id)
            image_paths.append(os.path.join(output_dir, f"a01-000u-00-{last_id:02d}.png"))


    # Crée le rapport des images (fonction commentée ici, vous pouvez la décommenter si nécessaire)
    report_path = os.path.join(output_dir, "rapport_images.txt")
    main_txt(search_inputs)

    # Déplace les images concaténées dans le dossier 'images'
    for img_path in image_paths:
        shutil.move(img_path, os.path.join(images_dir, os.path.basename(img_path)))
    #fusion rapport
    merge_reports(output_dir)
    #bouger images

# Exécuter l'automatisation
automate_image_combination()