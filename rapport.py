import re
import os
import shutil
def extraire_dernier_groupe(path):
    # Liste toutes les images dans le dossier
    images = lister_images(path)
    derniers_groupes = []

    # Recherche du dernier groupe de chiffres dans chaque image
    for image_nom in images:
        # Utilise une expression régulière pour extraire le dernier groupe de chiffres avant .png
        match = re.search(r'-(\d+)\.png$', image_nom)
        if match:
            derniers_groupes.append(int(match.group(1)))

    # Retourne le plus grand groupe de chiffres trouvé
    if derniers_groupes:
        return max(derniers_groupes)
    else:
        return 00

def lister_images(path):
    # Vérifie si le chemin est valide
    if not os.path.exists(path) or not os.path.isdir(path):
        raise ValueError(f"Le chemin spécifié n'existe pas ou n'est pas un dossier : {path}")
    
    # Crée une liste pour stocker les chemins des images
    images_paths = []
    
    # Parcourt tous les fichiers dans le dossier
    for root, dirs, files in os.walk(path):
        for file in files:
            # Vérifie si le fichier est une image avec des extensions courantes
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Ajoute le chemin complet de l'image à la liste
                images_paths.append(os.path.join(root, file))
    
    return images_paths


def deplacer_images(source_path, destination_path):
    # Vérifie si le dossier de destination existe, sinon le crée
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        print(f"Dossier créé : {destination_path}")

    images = lister_images(source_path)
    for image in images:
        image_source = os.path.join(source_path, image)
        image_destination = os.path.join(destination_path, image)
        
        # Déplace l'image
        shutil.move(image_source, image_destination)
        print(f"Image déplacée : {image} vers {destination_path}")

def concat_reports(report_dir, output_file):
    """Concatène toutes les lignes des rapports sauf celles commençant par '#' (sauf la première occurrence de début)
       et les enregistre dans un fichier de sortie unique."""
    
    # Liste pour stocker toutes les lignes non commentées
    all_lines = []
    first_occurrence = True  # Flag pour gérer la première ligne (de début)
    
    # Parcours tous les fichiers dans le dossier 'report_dir'
    for filename in os.listdir(report_dir):
        file_path = os.path.join(report_dir, filename)
        
        # Vérifie que le fichier est un fichier texte
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # Si la ligne commence par '#', on l'ignore, sauf si c'est la première ligne
                    if line.startswith('#'):
                        if first_occurrence:
                            all_lines.append(line)  # Ajouter la première ligne commentée
                            first_occurrence = False  # Désactiver le flag
                    else:
                        all_lines.append(line)  # Ajouter les autres lignes non commentées

    # Écriture des lignes concaténées dans le fichier de sortie
    with open(output_file, 'w', encoding='utf-8') as output_file:
        output_file.writelines(all_lines)