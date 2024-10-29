import os
import numpy as np
from matchingfiles import *
from PIL import Image, ImageChops, ImageFilter
import itertools
import random
from skimage import morphology, measure
from txt import *
import matplotlib.pyplot as plt

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

def concat_images(images, output_path):
    """Concatène les images données en une seule image juxtaposée et les sauvegarde à output_path."""
    opened_images = [Image.open(img).convert('RGB') for img in images]  # Convertir chaque image en RGB
    #print("Images chargées")

    # Recadrer et redimensionner les images pour enlever les bordures blanches ou transparentes
    processed_images = []
    target_height = 32  # Hauteur cible pour toutes les images
    for image in opened_images:
        trimmed_image = trim_image(image)  # Recadre l'image
        # Redimensionner chaque image à la même hauteur
        aspect_ratio = trimmed_image.width / trimmed_image.height
        new_width = int(target_height * aspect_ratio)
        resized_image = trimmed_image.resize((new_width, target_height), Image.LANCZOS)
        processed_images.append(resized_image)  # Ajoute l'image traitée à la liste

    #print("Images recadrées et redimensionnées")

    # Créer une nouvelle image ayant la largeur totale et la hauteur cible
    total_width = sum(img.width for img in processed_images)  # Largeur totale sans espaces
    combined_image = Image.new('RGB', (total_width, target_height), (255, 255, 255))  # Fond blanc
    #print(f"Nouvelle image créée avec largeur totale {total_width} et hauteur {target_height}")

    # Coller chaque image l'une à côté de l'autre sans espace
    x_offset = 0
    for img in processed_images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width  # Ajoute uniquement la largeur de l'image suivante
        #print(f"Image collée à l'offset {x_offset}")

    # Sauvegarder l'image combinée
    combined_image.save(output_path)
    #print(f"Image combinée sauvegardée dans {output_path}")

# Exemple d'utilisation
def automate_image_combination():
    matching_files, search_inputs = main()

    # Transformer le dictionnaire en une liste de listes
    lists_of_files = [matching_files[term] for term in matching_files]

    # Vérifier si au moins deux dossiers contiennent des images
    if len(lists_of_files) < 2:
        #print("Erreur : Au moins deux dossiers avec des images sont nécessaires.")
        return

    # Générer toutes les combinaisons possibles
    all_combinations = list(itertools.product(*lists_of_files))

    # Limiter à 100 combinaisons aléatoires si plus de 100 sont générées
    if len(all_combinations) > 100:
        selected_combinations = random.sample(all_combinations, 100)
    else:
        selected_combinations = all_combinations

    output_dir = get_output_path(search_inputs)

    counter = 1

    for combination in selected_combinations:
        images_to_concat = list(combination)
        output_file_name = f"combined_image_{''.join(search_inputs)}_{counter}.png"
        counter += 1

        output_path = os.path.join(output_dir, output_file_name)
        concat_images(images_to_concat, output_path)
    main_txt(search_inputs)

# Exécuter l'automatisation
automate_image_combination()



