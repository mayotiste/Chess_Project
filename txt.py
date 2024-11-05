import os
from PIL import Image
import numpy as np

def get_image_info(image_path, threshold=128):
    """Récupère des informations sur une image, telles que la taille, le niveau de gris, le format, et le seuil de binarisation."""
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

    # Calculer le niveau de gris moyen
    gray_level = np.mean(np.array(gray_image))

    return size, is_grayscale, img_format, threshold, gray_level

def create_image_report(image_paths, output_txt_path, search_terms):
    """Crée un rapport sous forme de fichier texte répertoriant les informations sur les images."""
    
    # En-tête du rapport
    header = (
        "#--- words.txt ---------------------------------------------------------------#\n"
        "#\n"
        "# iam database word information\n"
        "#\n"
        "# format: a01-000u-00-00 ok 154 1 408 768 27 51 AT A\n"
        "#\n"
        "#     a01-000u-00-00  -> word id for line 00 in form a01-000u\n"
        "#     ok              -> result of word segmentation\n"
        "#                            ok: word was correctly\n"
        "#                            er: segmentation of word can be bad\n"
        "#\n"
        "#     154             -> graylevel to binarize the line containing this word\n"
        "#     1               -> number of components for this word\n"
        "#     408 768 27 51   -> bounding box around this word in x,y,w,h format\n"
        "#     AT              -> the grammatical tag for this word, see the\n"
        "#                        file tagset.txt for an explanation\n"
        "#     A               -> the transcription for this word\n"
        "#\n"
    )

    # Ouvrir le fichier pour écrire les informations
    with open(output_txt_path, 'w') as report_file:
        # Écrire l'en-tête
        report_file.write(header)
        
        # Parcourir chaque image dans la liste des chemins d'images
        for index, image_path in enumerate(image_paths):
            # Récupérer les informations de l'image
            size, is_grayscale, img_format, threshold, gray_level = get_image_info(image_path)


            # Format de la ligne
            word_id = f"a01-000u-00-{index:02d}"  # Générer l'ID de mot
            result = "ok"  # Remplacer par une logique pour déterminer le résultat si nécessaire
            graylevel = threshold  # Utiliser le seuil comme niveau de gris
            components = 1  # Remplacer par le nombre réel de composants
            bounding_box = f"{size[0]} {size[1]} 27 51"  # Exemple de format (x, y, w, h)
            grammatical_tag = "AT"  # Exemple de balise grammaticale
            transcription = "A"  # Exemple de transcription

            # Ajouter les caractères de search_terms à la fin de la ligne sans espaces
            search_terms_str = ''.join(search_terms)  # Convertir la liste en chaîne sans espaces
            report_line = f"{word_id} {result} {graylevel} {components} {bounding_box} {grammatical_tag} {transcription} {search_terms_str}\n"

            # Écrire la ligne dans le fichier texte
            report_file.write(report_line)

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
    base_path = r"S:\\Base_de_donnees_concat\\image_concatenee"
    
    # Collecter les chemins d'images
    directories = [os.path.join(base_path, ''.join(search_terms))]
    
    image_paths = collect_images_from_directories(directories)
    
    # Générer le fichier de rapport des images
    output_txt_path = os.path.join(base_path, f"rapport_images_{''.join(search_terms)}.txt")
    create_image_report(image_paths, output_txt_path, search_terms)

# Lancer le programme principal
if __name__ == "__main__":
    # Par exemple, passez des termes de recherche comme une liste de chaînes de caractères
    search_terms = ["example"]
    main_txt(search_terms)
