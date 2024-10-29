import os

def list_subdirectories(parent_directory):
    # Vérifier si le répertoire parent existe
    if not os.path.exists(parent_directory):
        print(f"Le répertoire {parent_directory} n'existe pas.")
        return []

    # Lister les sous-dossiers
    subdirectories = [os.path.join(parent_directory, d) for d in os.listdir(parent_directory)
                      if os.path.isdir(os.path.join(parent_directory, d))]

    return subdirectories

# Spécifier le répertoire principal
parent_directory = "C:\\Users\\Utilisateur\\OneDrive\\Documents\\Chess\\BaseTestsReelle"  # Remplace par le chemin de ton dossier principal

# Obtenir la liste des sous-dossiers
subdirectories = list_subdirectories(parent_directory)

# Afficher les sous-dossiers
print("Sous-dossiers trouvés :")
for subdir in subdirectories:
    print(subdir)

    
