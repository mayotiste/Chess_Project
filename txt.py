import os
from PIL import Image
import numpy as np
from rapport import extraire_dernier_groupe
def get_image_info(image_path, threshold=128):
    """Récupère des informations sur une image, telles que la taille, le niveau de gris, le format, et le seuil de binarisation."""
    with Image.open(image_path) as img:
        size = img.size  # (largeur, hauteur)
        img_format = img.format
        is_grayscale = img.mode == 'L'
        gray_image = img.convert("L")
        gray_level = np.mean(np.array(gray_image))
    
    return size, is_grayscale, img_format, threshold, gray_level

def create_image_report(image_paths, base_path, report_path, search_terms, start_index=0):
    """Crée un contenu de rapport pour chaque ensemble de 100 images avec le nom du fichier comme ID."""
    header = (
        "#--- words.txt ---------------------------------------------------------------#\n"
        "#\n"
        "# iam database word information\n"
        "#\n"
        "# format: a01-000u-00-00 ok 154 1 408 768 27 51 AT A\n"
        "#\n"
    )
    
    report_content = header
    
    # Itérer sur les images à partir de start_index pour traiter 100 images
    for index, image_path in enumerate(image_paths[start_index:start_index+100], start=start_index):
        # Récupérer les informations sur l'image
        size, _, img_format, threshold, gray_level = get_image_info(image_path)
        
        # Utiliser le nom du fichier comme word_id (sans le chemin ni l'extension)
        word_id = os.path.basename(image_path).split('.')[0]
        
        nb_components = len(search_terms)
        result = "ok"
        bounding_box = calculate_bounding_box(image_path)
        grammatical_tag = "AT"
        search_terms_str = ''.join(search_terms)
        
        # Créer la ligne du rapport pour cette image
        report_line = (
            f"{word_id} {result} {int(gray_level)} {nb_components} {bounding_box} "
            f"{grammatical_tag} {search_terms_str}\n"
        )
        report_content += report_line

    # Créer ou fusionner le rapport
    with open(report_path, 'w') as report_file:
        report_file.write(report_content)
    print(f"Rapport généré à {report_path}")

def collect_images_from_directory(directory):
    """Collecte tous les chemins d'images dans le répertoire donné, triés par nom."""
    image_paths = []
    if os.path.exists(directory):
        for file_name in sorted(os.listdir(directory)):
            file_path = os.path.join(directory, file_name)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(file_path)
    return image_paths

def main_txt(search_terms):
    base_path = r"C:\Users\chess2425\Downloads\Chess_Project-Mael\Chess_Project-Mael\image_concatenee"
    image_paths = collect_images_from_directory(base_path)

    for i in range(0, len(image_paths), 100):
        output_txt_path = os.path.join(base_path, f"rapport_images_{''.join(search_terms)}_{i//100}.txt")
        create_image_report(image_paths,base_path, output_txt_path, search_terms, start_index=i)

if __name__ == "__main__":
    search_terms = ["example"]
    main_txt(search_terms)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def calculate_bounding_box(image_path, threshold=128):
    """Calcule la bounding box d'une image basée sur un seuil de binarisation."""
    """Calcule et affiche la bounding box d'une image avec l'image."""
    with Image.open(image_path) as img:
        # Convertir l'image en niveaux de gris
        gray_image = img.convert("L")
        
        # Convertir en tableau numpy
        img_array = np.array(gray_image)
        
        # Appliquer un seuil pour obtenir une image binaire
        binary_image = img_array < threshold
        
        # Trouver les positions des pixels non nuls (True)
        coords = np.argwhere(binary_image)
        
        # Si des pixels sont trouvés, calculer la bounding box
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bounding_box = (x_min, y_min, x_max, y_max)
        else:
            # Si aucun pixel n'est trouvé, retourner une bounding box vide
            bounding_box = (0, 0, 0, 0)
        
        # Afficher l'image avec la bounding box
        #plt.imshow(img, cmap="gray")
        # Dessiner la bounding box
        # plt.gca().add_patch(
        #     plt.Rectangle(
        #         (x_min, y_min),  # Position (x, y)
        #         x_max - x_min,   # Largeur
        #         y_max - y_min,   # Hauteur
        #         edgecolor="red",
        #         facecolor="none",
        #         linewidth=2
        #     )
        # )
        # plt.title("Image avec Bounding Box")
        # plt.axis("off")  # Masquer les axes
        # plt.close()
    
    return bounding_box

