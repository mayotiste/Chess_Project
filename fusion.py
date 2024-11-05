import os

def fusionner_fichiers_txt(fichiers, fichier_sortie):
    with open(fichier_sortie, 'w', encoding='utf-8') as f_sortie:
        for index, fichier in enumerate(fichiers):
            # Vérifiez que le fichier est un fichier texte (.txt)
            if fichier.endswith('.txt'):
                with open(fichier, 'r', encoding='utf-8') as f_texte:
                    for ligne in f_texte:
                        # Pour le premier fichier, écrivez toutes les lignes
                        if index == 0:
                            f_sortie.write(ligne)
                        # Pour les fichiers suivants, écrivez les lignes qui ne commencent pas par '#'
                        else:
                            if not ligne.startswith('#'):
                                f_sortie.write(ligne)

# Liste des fichiers à fusionner (indiquez les chemins complets)
fichiers_a_fusionner = [
    "S:\\Base_de_donnees_concat\\image_concatenee\\rapport_images_Rxe4.txt",
    "S:\\Base_de_donnees_concat\\image_concatenee\\rapport_images_Fh6.txt"
]

# Spécifiez le chemin où vous souhaitez enregistrer le fichier combiné
fichier_combined = "S:\\Base_de_donnees_concat\\image_concatenee\\words.txt"

# Appel de la fonction
fusionner_fichiers_txt(fichiers_a_fusionner, fichier_combined)
