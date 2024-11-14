import os

def merge_reports(base_path):
    """Fusionne les rapports 'rapport_images' dans un rapport global, écrasant l'existant."""
    rapport_global_path = os.path.join(r'C:\Users\chess2425\Downloads\Chess_Project-Mael\Chess_Project-Mael', "rapport_global.txt")

    # Lister les fichiers dans le répertoire
    report_files = [f for f in os.listdir(base_path) if f.startswith("rapport_images")]

    # Liste pour stocker les lignes des rapports
    all_lines = []

    # Parcourir chaque fichier de rapport trouvé
    for report_file in report_files:
        report_path = os.path.join(base_path, report_file)
        
        # Lire le fichier rapport sans les lignes commençant par '#'
        with open(report_path, 'r') as file:
            lines = file.readlines()
            # Ajouter les lignes sans celles qui commencent par '#'
            for line in lines:
                if not line.startswith('#'):
                    all_lines.append(line)

    # Sauvegarder toutes les lignes dans rapport_global.txt, écrasant l'existant
    with open(rapport_global_path, 'w') as global_file:
        global_file.writelines(all_lines)
    
    print(f"Rapport global généré ou mis à jour à {rapport_global_path}")