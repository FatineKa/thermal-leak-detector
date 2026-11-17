# télécharge un échantillon au hasard des dpe (diagnostics énergie) depuis l'api de l'ademe
# à lancer depuis le dossier du projet :  python scripts/download_data.py
# résultat : data/dpe_sample.parquet
# autre taille :  python scripts/download_data.py --nb-lignes 2000000

import argparse
import time

import pandas as pd
import requests      # pour demander des données sur internet (http requests)

# adresse de l'api (api url)
adresse = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"

# réglages qu'on peut changer au lancement (arguments)
parser = argparse.ArgumentParser()
parser.add_argument("--nb-lignes", type=int, default=300_000)  # nombre de logements à garder (rows)
parser.add_argument("--sortie", default="data/dpe_sample.parquet")  # fichier de résultat
args = parser.parse_args()

nb_lignes_voulues = args.nb_lignes
taille_page = 10_000  # nombre de logements par demande (page size)

# les colonnes à télécharger (columns)
colonnes = [
    # pour enlever les doublons (duplicates)
    "numero_dpe",
    "numero_dpe_remplace",
    "date_etablissement_dpe",

    # la réponse à deviner : la classe de a à g (target)
    "etiquette_dpe",

    # ce qu'un locataire peut savoir sans diagnostic (features)
    "surface_habitable_logement",
    "annee_construction",
    "periode_construction",
    "type_batiment",                       # maison ou appartement
    "type_energie_principale_chauffage",   # électricité, gaz, fioul, bois...
    "type_installation_chauffage",         # individuel ou collectif
    "code_departement_ban",
    "zone_climatique",
    "classe_altitude",                     # montagne ou pas
    "numero_etage_appartement",            # étage (floor)
    "position_logement_dans_immeuble",     # bas, milieu ou haut de l'immeuble
    "nombre_niveau_logement",              # nombre d'étages du logement
    "typologie_logement",                  # t1, t2, t3...
    "hauteur_sous_plafond",                # hauteur sous plafond (ceiling height)
    "type_energie_principale_ecs",         # énergie de l'eau chaude (hot water)
    "type_generateur_chauffage_principal", # chaudière, pompe à chaleur, radiateur...
    "presence_production_pv",              # panneaux solaires oui ou non (solar panels)
]

# sort=_rand : l'api donne les logements dans un ordre au hasard, toujours le même.
# les 300 000 premiers forment donc un échantillon au hasard de toute la france,
# identique à chaque lancement.
lien = adresse
parametres = {"size": taille_page, "sort": "_rand", "select": ",".join(colonnes)}
pages = []
nb_lignes = 0

while nb_lignes < nb_lignes_voulues:
    # sur des centaines de demandes, une peut échouer : on réessaie jusqu'à 3 fois
    for essai in range(3):
        try:
            reponse = requests.get(lien, params=parametres, timeout=120)
            reponse.raise_for_status()
            break
        except requests.RequestException:
            if essai == 2:
                raise
            time.sleep(10)
    donnees = reponse.json()
    pages.append(pd.DataFrame(donnees["results"]))
    nb_lignes += len(donnees["results"])
    print(f"{nb_lignes:,} logements téléchargés")
    lien = donnees["next"]  # l'api donne le lien de la page suivante (next page)
    parametres = None       # ce lien contient déjà les paramètres

# colle toutes les pages ensemble, garde le bon nombre de lignes, et enregistre le fichier
tableau = pd.concat(pages).head(nb_lignes_voulues)[colonnes]
tableau.to_parquet(args.sortie, index=False)
print(f"{len(tableau):,} logements enregistrés dans {args.sortie}")
