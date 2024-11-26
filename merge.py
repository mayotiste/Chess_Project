import os

import os

def merge_reports(base_path):
    """Fusionne les rapports 'rapport_images' dans un rapport global, écrasant l'existant."""
    
    # Header à ajouter au début du fichier
    header = (
        "#--- words.txt ---------------------------------------------------------------#\n"
        "# iam database word information\n"
        "#\n"
        "# format: a01-000u-00-00 ok 154 1 408 768 27 51 AT A\n"
        "#\n"
        "#     a01-000u-00-00  -> word id for line 00 in form a01-000u\n"
        "#     ok              -> result of word segmentation\n"
        "#                            ok: word was correctly\n"
        "#                            er: segmentation of word can be bad\n"
        "#     154             -> graylevel to binarize the line containing this word\n"
        "#     1               -> number of components for this word\n"
        "#     408 768 27 51   -> bounding box around this word in x,y,w,h format\n"
        "#     AT              -> the grammatical tag for this word, see the\n"
        "#                        file tagset.txt for an explanation\n"
        "#     A               -> the transcription for this word\n"
    )
    
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
    
    # Sauvegarder le header et toutes les lignes dans rapport_global.txt, écrasant l'existant
    with open(rapport_global_path, 'w') as global_file:
        global_file.write(header)  # Ajouter le header en premier
        global_file.writelines(all_lines)  # Ajouter les lignes des rapports
    
    print(f"Rapport global généré ou mis à jour à {rapport_global_path}")

