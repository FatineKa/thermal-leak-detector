# télécharge un échantillon au hasard des dpe (diagnostics énergie) depuis l'api de l'ademe
# à lancer depuis le dossier du projet :  python scripts/download_data.py
# résultat : data/dpe_sample.parquet

import pandas as pd
import requests      # pour demander des données sur internet (http requests)

# adresse de l'api (api url)
adresse = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"

nb_lignes_voulues = 300_000  # nombre de logements à garder (rows)
taille_page = 10_000         # nombre de logements par demande (page size)

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
    reponse = requests.get(lien, params=parametres, timeout=120)
    reponse.raise_for_status()
    donnees = reponse.json()
    pages.append(pd.DataFrame(donnees["results"]))
    nb_lignes += len(donnees["results"])
    print(f"{nb_lignes:,} logements téléchargés")
    lien = donnees["next"]  # l'api donne le lien de la page suivante (next page)
    parametres = None       # ce lien contient déjà les paramètres

# colle toutes les pages ensemble, garde 300 000 lignes, et enregistre le fichier
tableau = pd.concat(pages).head(nb_lignes_voulues)[colonnes]
tableau.to_parquet("data/dpe_sample.parquet", index=False)
print(f"{len(tableau):,} logements enregistrés dans data/dpe_sample.parquet")
