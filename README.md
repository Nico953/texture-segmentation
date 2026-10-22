Segmentation d'images texturees par filtrage de Gabor et dimension fractale
Projet de Traitement de l'image  Master 1 Systemes Intelligents, Sorbonne Universite.
Objectif
Ce projet etudie la segmentation non supervisee d'images texturees en s'inspirant de l'article « Segmentation of textured images based on fractals and image filtering » (Kasparis et al., Pattern Recognition, 2001).

L'objectif est de regrouper les pixels selon leurs textures, sans utiliser la verite terrain pendant l'apprentissage, puis de mesurer la qualite des segmentations obtenues.
Methode
Le pipeline comprend les etapes suivantes :
1. Filtrage de Gabor : extraction de 12 reponses (4 orientations et 3 frequences).
2. Caracterisation des textures : calcul d'une dimension fractale locale sur les reponses filtrees.
3. Selection des caracteristiques : score F calcule sur des blocs de 16 x 16 pixels, avec conservation des caracteristiques dont le score est superieur ou egal a Fmax / 4.5.
4. Segmentation : comparaison d'un K-means standard et d'une variante iterative utilisant des voisinages de 33 x 33, 25 x 25 et 17 x 17 pixels.
5. Evaluation : calcul du pourcentage de pixels correctement classes (PCC) apres alignement optimal des etiquettes des clusters avec la verite terrain.
La variante iterative reprend le principe de raffinement progressif de l'article, avec des choix d'implementation propres au projet ; elle ne constitue pas necessairement une reproduction exacte de l'algorithme publie.
Donnees
Les experiences utilisent des textures de Brodatz, assemblees en mosaiques de quatre regions (2 x 2). Chaque texture est recadree au centre sur 128 x 128 pixels, donnant une mosaique de 256 x 256 pixels.

La mosaique utilisee dans l'experience principale contient les textures D9, D68, D84 et D112. Les fichiers images doivent etre places dans data/brodatz/ ; ce repertoire est ignore par Git dans la configuration actuelle.
Installation
Python 3 avec les dependances declarees dans requirements.txt est necessaire.

Depuis la racine du projet :
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Sous macOS ou Linux, activer l'environnement avec source venv/bin/activate.
Execution
Depuis la racine du projet :
python src/experiment_brodatz.py
Cette commande affiche la mosaique, la verite terrain, les segmentations standard et iterative, ainsi que leurs PCC.

Pour comparer la dimension fractale a l'energie des reponses de Gabor sous bruit multiplicatif gaussien :
python src/noise_brodatz.py
Ce second script utilise le K-means standard pour les deux types de caracteristiques, avec cinq repetitions par niveau de bruit (0, 0.1, 0.2, 0.4). Il affiche les moyennes et ecarts-types du PCC.
Resultats sur la mosaique D9 / D68 / D84 / D112
Comparaison des methodes de segmentation
Methode	PCC
K-means standard	91,98 %
K-means iteratif	95,29 %
Difference	+3,31 points


Robustesse au bruit multiplicatif
Niveau de bruit	Dimension fractale (PCC moyen ± ecart-type)	Energie de Gabor (PCC moyen ± ecart-type)
0,0	91,98 ± 0,01 %	55,18 ± 0,01 %
0,1	81,38 ± 1,31 %	54,96 ± 0,22 %
0,2	85,50 ± 0,93 %	54,51 ± 0,33 %
0,4	81,47 ± 3,21 %	53,69 ± 0,59 %


Sur cette mosaique, la dimension fractale donne une segmentation nettement plus precise que l'energie de Gabor. Son PCC ne diminue toutefois pas de maniere strictement monotone lorsque le bruit augmente. Ces resultats sont propres a la mosaique et aux parametres testes ; ils ne constituent pas une evaluation globale de la base Brodatz.

La selection automatique des caracteristiques peut retenir un nombre different de composantes selon la representation (12 pour la dimension fractale et 8 pour l'energie sur cette mosaique). La comparaison porte donc sur les pipelines complets, et non sur deux ensembles de caracteristiques de meme dimension.
Organisation du code
Fichier	Role
src/gabor.py	Construction et application des filtres de Gabor
src/fractal.py	Extraction des caracteristiques fractales
src/feature_selection.py	Calcul du score F et selection
src/clustering.py	K-means standard, variante iterative et PCC
src/brodatz.py	Chargement des textures et creation des mosaiques
src/evaluate.py	Ajout de bruit et pipeline d'evaluation
src/experiment_brodatz.py	Comparaison des deux segmentations
src/noise_brodatz.py	Experience de robustesse au bruit
notebooks/demo.ipynb	Notebook de demonstration


Reference
Kasparis et al., « Segmentation of textured images based on fractals and image filtering », Pattern Recognition, 2001.