import os
from PIL import Image
import numpy as np

def get_image_info(image_path, threshold=128):
    """Récupère des informations sur une image, telles que la taille, le niveau de gris, le format, et le seuil de binarisation."""
     # Partie fixe que l'on souhaite inclure dans chaque fichier texte
    fixed_content = (
        "===== RAPPORT D'ANALYSE D'IMAGES =====\n"
        "Ce fichier contient des informations sur les images analysées.\n"
        "Les paramètres répertoriés incluent :\n"
        "- Seuil de binarisation\n"
        "- Taille (largeur x hauteur)\n"
        "- Niveau de gris moyen\n"
        "- Format de l'image\n"
        "======================================\n\n"
    )
    # Ouvrir l'image
    with Image.open(image_path) as img:
        # Obtenir la taille et le format
        size = img.size  # (largeur, hauteur)
        img_format = img.format
        
        # Vérifier si l'image est en niveaux de gris
        is_grayscale = img.mode == 'L'  # True si image en niveaux de gris, sinon False
        
        # Convertir l'image en niveaux de gris pour la binariser (si elle ne l'est pas déjà)
        gray_image = img.convert("L")
        
        # Binariser l'image avec le seuil donné
        binary_image = gray_image.point(lambda p: p > threshold and 255)
        
    return size, is_grayscale, img_format, threshold

def create_image_report(image_paths, output_txt_path):
    """Crée un rapport sous forme de fichier texte répertoriant les informations sur les images."""
     # Partie fixe que l'on souhaite inclure dans chaque fichier texte
    fixed_content = (
        "===== RAPPORT D'ANALYSE D'IMAGES =====\n"
        "Ce fichier contient des informations sur les images analysées.\n"
        "Les paramètres répertoriés incluent :\n"
        "- Seuil de binarisation\n"
        "- Taille (largeur x hauteur)\n"
        "- Niveau de gris moyen\n"
        "- Format de l'image\n"
        "======================================\n\n"
    )
    
    # Ouvrir le fichier pour écrire les informations
    with open(output_txt_path, 'w') as report_file:
        # Parcourir chaque image dans la liste des chemins d'images
        for image_path in image_paths:
            # Récupérer les informations de l'image
            size, is_grayscale, img_format, threshold = get_image_info(image_path)
            
            # Écrire les informations dans le fichier texte
            report_file.write(f"Image: {os.path.basename(image_path)}\n")
            report_file.write(f" - Taille: {size[0]}x{size[1]}\n")
            report_file.write(f" - Niveau de gris: {'Oui' if is_grayscale else 'Non'}\n")
            report_file.write(f" - Format: {img_format}\n")
            report_file.write(f" - Seuil de binarisation: {threshold}\n")
            report_file.write("\n")  # Ajouter une ligne vide entre chaque image

    print(f"Rapport généré à {output_txt_path}")

def collect_images_from_directories(directories):
    """Collecte tous les chemins d'images dans les répertoires donnés."""
    image_paths = []
    for directory in directories:
        if os.path.exists(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                # Vérifier que c'est bien une image avant de l'ajouter
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_paths.append(file_path)
    return image_paths

# Exécution principale
def main_txt(search_terms):
    # Liste des répertoires où se trouvent les images
    
    # Collecter les chemins d'images
    directories = [f"S:\Base de données concat\image_concatenee\{''.join(search_terms)}"]
    
    image_paths = collect_images_from_directories(directories)
    
    # Générer le fichier de rapport des images
    output_txt_path = os.path.join(directories[0], f"rapport_images_{''.join(search_terms)}.txt")
    create_image_report(image_paths, output_txt_path)

# Lancer le programme principal
if __name__ == "__main__":
    main_txt()