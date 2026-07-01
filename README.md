# LeWM, modèle du monde latent pour le stationnement

Apprentissage d'un modèle du monde en espace latent sur l'environnement de
stationnement `parking-v0` de highway-env, puis planification dans ce latent.
Un encodeur transforme la vue de dessus en un vecteur latent, un prédicteur
conditionné par l'action prédit le latent au pas suivant, et un planificateur
cherche la séquence d'actions qui rapproche le latent du latent visé.

## Méthode

Encodeur: un ViT, image 64x64 vers un latent de dimension 192.

Prédicteur: un transformer conditionné par l'action (accélération, direction)
qui prédit le latent au pas suivant, entraîné par une perte de prédiction en
espace latent.

Régularisation: SIGReg (Balestriero et LeCun, LeJEPA, 2025) pousse la
distribution des latents vers une gaussienne isotrope. On y ajoute un terme de
variance et de covariance (VICReg) qui empêche l'effondrement des latents.

Planification: recherche par entropie croisée (CEM) sur des séquences d'actions,
chaque séquence étant évaluée par la distance entre le latent prédit en fin de
séquence et le latent visé, en horizon fuyant.

Stationnement de référence: une manœuvre analytique sert de contrôleur
privilégié fiable, chemin de Dubins à courbure bornée vers une pose d'entrée
alignée sur la place, suivi en poursuite de point, puis créneau terminal.

## L'effondrement latent

Le régularisateur SIGReg standardise les latents avant son test de gaussianité,
ce qui le rend invariant à l'échelle: il ne pénalise donc pas l'effondrement.
Sans autre contrainte, l'encodeur envoie toutes les images sur le même point, le
coût de planification devient plat, et le planificateur n'agit plus. Le terme de
variance et de covariance corrige ce défaut. Après correction, l'écart-type par
dimension des latents passe d'environ 0.05, cas effondré, à environ 1.0, et le
planificateur produit de nouveau des actions utiles.

## Résultats

Effondrement corrigé: écart-type par dimension des latents proche de 1.0 pendant
tout l'entraînement, perte de prédiction en baisse.

Stationnement, contrôleur analytique de référence: 100% de réussite stricte sur
10 graines, distance finale d'environ 1.7 cm au centre de la place, environ 0.9°
d'écart à l'axe, aucune collision.

Planificateur latent du modèle: conduit la voiture vers la place mais n'atteint
pas de façon fiable l'alignement précis. C'est la limite actuelle, avec un jeu
d'entraînement modeste d'environ 2250 épisodes.

## Données

Environnement `parking-v0` de highway-env. Les trajectoires sont collectées par
plusieurs politiques et stockées en HDF5, image, actions, image de la place
visée, et métadonnées par épisode. Le rechunk des observations par frame
accélère fortement la lecture aléatoire pendant l'entraînement.

## Installation

Voir SETUP.md. En résumé:

    python -m venv .venv
    .venv/bin/pip install torch gymnasium highway-env h5py pygame imageio numpy

## Utilisation

Collecte puis rechunk des données:

    python scripts/collect_parking.py --episodes 2500 --out data/parking/train_v2.h5
    python scripts/repack_h5.py --src data/parking/train_v2.h5 --dst data/parking/train_v2_fast.h5

Entraînement:

    python scripts/train.py --data data/parking/train_v2_fast.h5 --out runs/lewm --epochs 4 --device mps

Évaluation du stationnement:

    python scripts/eval_parking.py --planner biarc --episodes 10
    python scripts/eval_parking.py --planner model_warm --episodes 10

Visualisation temps réel, fenêtre unique avec les trajectoires planifiées
superposées à la scène:

    python scripts/imagine_viewer.py --drive biarc          # se gare entre les lignes
    python scripts/imagine_viewer.py --drive model --warm   # trajectoires imaginées par le modèle

## Structure

    src/       encodeur, prédicteur, régularisation, environnement, métriques, contrôleur analytique
    scripts/   collecte, entraînement, planification, évaluation, visualisation
    tests/     tests unitaires

## Références

- Balestriero, LeCun. LeJEPA, arXiv:2511.08544, 2025. SIGReg.
- Bardes, Ponce, LeCun. VICReg, 2022. Terme de variance et covariance.
- Dubins. On curves of minimal length with a constraint on average curvature, 1957.
- highway-env, environnement parking-v0.
