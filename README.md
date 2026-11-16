# thermal-leak-detector

Estimate a home's French energy rating (DPE, A to G) from simple facts, without a diagnosis.

## Data

- DPE existing homes (since July 2021), ADEME, data.gouv.fr
- 300,000 homes sampled at random, all of France
- 276,842 after cleaning: 80% training, 20% test

## Cleaning

- whole buildings removed
- living area: 9 to 500 m² (9 m²: legal minimum)
- ceiling height: 2.2 to 5 m
- heating equipment: 130 types grouped into 7 families + unknown
- January 1st 2026 rule change: `apres_2026` column

## Excluded columns (answer leakage)

- energy consumption, CO₂ emissions, GHG label
- energy costs
- technical details: insulation, windows, heat losses

## Results

| | Baseline | Model |
|---|---|---|
| Exact class | 44.4% | 47.9% |
| Within one class | 86.4% | 86.5% |
| F1 macro | 0.28 | 0.42 |
| Passoire (F/G) recall | 24.7% | 43.6% |
| Passoire (F/G) precision | 52.6% | 44.9% |

- Baseline: most frequent class per (construction period × heating energy)
- Model: XGBoost in a scikit-learn pipeline, softened class weights, tuned hyperparameters

## Findings

- 100k → 221k training rows: +0.2 points accuracy
- Hyperparameter tuning: +0.7 F1 points
- The limit comes from the available information, not the amount of data or the model

## Stack

- AWS S3
- Amazon SageMaker AI
- Python: pandas, scikit-learn, XGBoost

## Run

```bash
python scripts/download_data.py
python scripts/clean_data.py
python src/train.py
python scripts/launch_training.py
```

---

# thermal-leak-detector

Estimer la classe DPE (A à G) d'un logement à partir d'informations simples, sans diagnostic.

## Données

- DPE logements existants (depuis juillet 2021), ADEME, data.gouv.fr
- 300 000 logements tirés au hasard, toute la France
- 276 842 après nettoyage : 80 % entraînement, 20 % test

## Nettoyage

- immeubles entiers retirés
- surface : 9 à 500 m² (9 m² : minimum légal)
- hauteur sous plafond : 2,2 à 5 m
- équipement de chauffage : 130 types regroupés en 7 familles + inconnu
- règle du 1er janvier 2026 : colonne `apres_2026`

## Colonnes exclues (fuite de la réponse)

- consommation d'énergie, émissions de CO₂, étiquette GES
- coûts d'énergie
- détails techniques : isolation, fenêtres, déperditions

## Résultats

| | Baseline | Modèle |
|---|---|---|
| Classe exacte | 44,4 % | 47,9 % |
| À une classe près | 86,4 % | 86,5 % |
| F1 macro | 0,28 | 0,42 |
| Passoire (F/G) rappel | 24,7 % | 43,6 % |
| Passoire (F/G) précision | 52,6 % | 44,9 % |

- Baseline : classe la plus fréquente par (période de construction × énergie de chauffage)
- Modèle : XGBoost dans un pipeline scikit-learn, poids de classes adoucis, hyperparamètres réglés

## Constats

- 100 000 → 221 000 lignes d'entraînement : +0,2 point d'exactitude
- Réglage des hyperparamètres : +0,7 point de F1
- La limite vient des informations disponibles, pas de la quantité de données ni du modèle

## Outils

- AWS S3
- Amazon SageMaker AI
- Python : pandas, scikit-learn, XGBoost

## Lancer

```bash
python scripts/download_data.py
python scripts/clean_data.py
python src/train.py
python scripts/launch_training.py
```
