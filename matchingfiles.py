import os

def find_matching_files(directories, search_terms):
    """Retourne un dictionnaire avec les chemins complets des fichiers correspondants pour chaque terme de recherche dans les répertoires donnés."""
    matching_files = {term: [] for term in search_terms}

    # Parcourir chaque répertoire
    for i, directory in enumerate(directories):
        # Vérifier si le répertoire existe
        if os.path.exists(directory):
            # Parcourir tous les fichiers du répertoire
            for file_name in os.listdir(directory):
                # Ajouter tous les fichiers du répertoire correspondant au terme
                term = search_terms[i]
                if term in directory:
                    # Ajouter le chemin complet du fichier à la liste pour ce terme
                    file_path = os.path.join(directory, file_name)
                    matching_files[term].append(file_path)
    
    return matching_files

def main():
    # Demander à l'utilisateur d'entrer une chaîne de caractères complète
    search_input = input("Entrez la chaîne de caractères (maximum 6) : ").strip()

    # Limiter l'entrée à un maximum de 6 caractères
    search_inputs = list(search_input[:6])  # Convertit la chaîne en liste de caractères

    if not search_inputs:
        print("Aucun caractère entré.")
        return

    # Liste des termes de recherche
    search_terms = search_inputs

    # Liste des répertoires contenant les images pour chaque terme de recherche
    directories = [f"S:\\BaseImages\\BaseTestsReelle//{term}" for term in search_terms]

    # Trouver les fichiers correspondants
    matching_files = find_matching_files(directories, search_terms)

    return matching_files, search_inputs
    

def get_output_path(search_inputs, base_dir="S:\\Base_de_donnees_concat\\image_concatenee"):
    """Retourne le chemin final où les images concaténées seront sauvegardées."""
    # Joindre les caractères entrés pour créer le nom du dossier
    output_folder_name = f"{''.join(search_inputs)}"
    
    # Créer le chemin complet en combinant le répertoire de base et le nom du dossier
    output_path = os.path.join(base_dir, output_folder_name)
    
    # Créer le dossier s'il n'existe pas déjà
    os.makedirs(output_path, exist_ok=True)
    
    return output_path