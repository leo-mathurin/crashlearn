# Circuits et partitions d'évaluation

La source unique est [`src/crashlearn/tracks.yaml`](../src/crashlearn/tracks.yaml), exposée par le module `crashlearn.tracks`. Aucune autre liste de cartes ne doit exister dans le dépôt : le code d'entraînement et d'évaluation passe par ce module.

```python
from crashlearn import tracks

tracks.train_tracks()  # 15 circuits
tracks.validation_tracks()  # 4 circuits
tracks.canonical_name("Mexico City")  # "MexicoCity"
tracks.final_retraining_tracks()  # 23 circuits, test compris
tracks.sealed_test_tracks(unseal=True)  # test scellé, voir la règle
```

`final_retraining_tracks()` est réservé au réentraînement final après le gel. Sans `unseal=True`, `sealed_test_tracks()` lève `PermissionError`.

## Inventaire

`vendor/simulation/maps/` contient 25 dossiers : 23 circuits distincts, le doublon `Mexico City` et `examples/`, la carte par défaut quand `set_map` n'est pas appelé. Chaque circuit fournit :

- `<Nom>_centerline.csv` : `x_m, y_m, w_tr_right_m, w_tr_left_m` ;
- `<Nom>_map.png` et `<Nom>_map.yaml` : grille d'occupation, résolution et origine ;
- `<Nom>_DonkeySim_waypoints.txt` ;
- `<Nom>_raceline.csv`, **sauf Montreal et Shanghai**.

Aucun `<Nom>_config.yaml` n'est fourni : le simulateur prend le premier point de la centerline comme pose de départ.

Le sha256 de chaque fichier est enregistré dans `tracks.yaml`. Deux commandes le contrôlent :
- `uv run python scripts/track_inventory.py` compare le YAML aux fichiers vendorisés et renvoie le code de sortie 1 s'ils divergent ;
- `--write` régénère uniquement les champs `assets` et `geometry_*`.

| Circuit | Partition | Raceline | Longueur (m) | Cap cumulé (rad) | κ p50 | κ p90 | κ max | Virages serrés |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Austin | train | oui | 421.0 | 33.4 | 0.008 | 0.226 | 1.31 | 11 |
| BrandsHatch | train | oui | 356.3 | 19.0 | 0.014 | 0.147 | 0.52 | 1 |
| Budapest | train | oui | 402.6 | 27.5 | 0.011 | 0.217 | 0.77 | 4 |
| Catalunya | train | oui | 416.8 | 29.3 | 0.012 | 0.189 | 1.05 | 5 |
| Hockenheim | train | oui | 359.8 | 23.7 | 0.011 | 0.226 | 1.10 | 4 |
| IMS | train | oui | 293.1 | 6.4 | 0.003 | 0.058 | 0.07 | 0 |
| Melbourne | train | oui | 474.3 | 25.6 | 0.011 | 0.132 | 1.41 | 8 |
| MexicoCity | train | oui | 356.7 | 26.4 | 0.004 | 0.206 | 1.32 | 13 |
| Monza | train | oui | 446.1 | 17.9 | 0.003 | 0.114 | 1.32 | 6 |
| Nuerburgring | train | oui | 446.1 | 29.1 | 0.010 | 0.227 | 0.84 | 3 |
| Sakhir | train | oui | 441.9 | 24.7 | 0.002 | 0.156 | 1.24 | 9 |
| SaoPaulo | train | oui | 344.7 | 25.8 | 0.026 | 0.214 | 0.86 | 7 |
| Sepang | train | oui | 487.0 | 30.8 | 0.006 | 0.193 | 1.02 | 10 |
| Silverstone | train | oui | 457.9 | 28.7 | 0.014 | 0.189 | 1.05 | 4 |
| Zandvoort | train | oui | 387.9 | 29.0 | 0.031 | 0.233 | 0.90 | 1 |
| Montreal | validation | **non** | 285.0 | 23.1 | 0.012 | 0.264 | 1.32 | 9 |
| Sochi | validation | oui | 463.8 | 28.6 | 0.011 | 0.170 | 1.73 | 13 |
| Spielberg | validation | oui | 343.3 | 17.4 | 0.008 | 0.153 | 1.58 | 4 |
| YasMarina | validation | oui | 398.0 | 33.7 | 0.007 | 0.243 | 1.85 | 18 |
| MoscowRaceway | test | oui | 322.8 | 30.3 | 0.007 | 0.321 | 1.25 | 12 |
| Oschersleben | test | oui | 260.7 | 23.9 | 0.005 | 0.293 | 0.70 | 6 |
| Shanghai | test | **non** | 497.6 | 33.5 | 0.004 | 0.168 | 1.73 | 9 |
| Spa | test | oui | 554.4 | 30.4 | 0.010 | 0.170 | 1.61 | 5 |

Les valeurs κ désignent la courbure absolue, en rad/m. Les chiffres exacts se trouvent dans `tracks.yaml`.

## Doublon `Mexico City`

`Mexico City/` est un doublon exact de `MexicoCity/`. Les 5 fichiers ont un sha256 identique, et ce test est rejoué par `track_inventory.py` et `tests/test_tracks.py`. Un seul nom de fichier diffère : `Mexico City_DonkeySim_waypoints.txt`. Le nom canonique retenu est `MexicoCity`, et `Mexico City` n'en est qu'un alias.

Deux défauts du simulateur, à corriger dans E-12 et laissés tels quels ici :
- `get_available_maps()` renvoie **24 noms**, alias compris. Il ne faut donc pas s'en servir pour lister les circuits.
- `set_map("Mexico City")` échoue au `reset` : il cherche `maps/Mexico City/Mexico City_map.png`, qui n'existe pas (`FileNotFoundError`). Tant que ce n'est pas corrigé, toujours passer par `canonical_name`.

## Partitions

Les partitions ont été fixées au cadrage, à partir d'un audit statique, puis confrontées à l'arborescence réelle : **aucun écart**, les 23 noms existent avec la même casse.

- **Train (15)** : Austin, BrandsHatch, Budapest, Catalunya, Hockenheim, IMS, Melbourne, MexicoCity, Monza, Nuerburgring, Sakhir, SaoPaulo, Sepang, Silverstone, Zandvoort.
- **Validation (4)** : Montreal, Sochi, Spielberg, YasMarina.
- **Test scellé (4)** : MoscowRaceway, Oschersleben, Shanghai, Spa.

Les métriques géométriques **décrivent** ce découpage, elles ne le justifient pas.
- La validation mêle :
  - un tracé court et dense (Montreal) ;
  - un tracé long (Sochi) ;
  - un tracé peu sinueux (Spielberg, 4 virages serrés) ;
  - le tracé le plus sinueux de l'inventaire (YasMarina, 18 virages serrés).
- Le test est volontairement exigeant :
  - deux tracés courts et denses (MoscowRaceway et Oschersleben, les deux κ p90 les plus élevés) ;
  - deux tracés longs (Shanghai et Spa, Spa étant le plus long de l'inventaire).

Les listes sont recopiées en dur dans `tests/test_tracks.py`. Toute modification de `tracks.yaml` qui déplace un circuit fait donc échouer la CI : **une partition modifiée après coup invalide la mesure de généralisation**, et ce changement relève d'une décision d'équipe.

## Règle du test scellé

- Le test n'est accessible que par `sealed_test_tracks(unseal=True)`. Aucune fonction par défaut ne le renvoie, sauf `final_retraining_tracks()`, réservée au réentraînement final.
- Il est ouvert **une seule fois**, après le gel du **17/01/2027**, pour mesurer le modèle gelé.
- **Aucun ajustement** (hyperparamètres, features, récompense, choix de checkpoint) ne doit ensuite reposer sur ses résultats.
- Une fois consulté, il perd définitivement son caractère inédit : une seconde mesure ne vaut plus comme mesure de généralisation.
- Ne pas lancer de politique sur ces cartes avant le gel, y compris « pour voir ». Le test de chargement (`check_track_loading.py`) se contente de rouler en ligne droite pendant 1 s, sans politique.

## Disponibilité des assets

Montreal (validation) et Shanghai (test) n'ont **pas de raceline**. Aujourd'hui, **aucun composant n'en dépend** :
- la progression et le simulateur n'utilisent que la centerline (`env_simulation.py`, `sim_recorder.py`) ;
- la raceline n'apparaît que dans les outils `maps/convert.py` et `maps/rename.py`.

Si une récompense, une feature ou le replay venait à utiliser la raceline, il faudrait prévoir un repli sur la centerline pour ces deux circuits.

`INSTRUCTIONS.md` cite Monaco en exemple, mais cette carte **n'est pas fournie**. Aucune configuration ne la référence, et `canonical_name("Monaco")` lève `KeyError`.

Chargement réel : `uv run python scripts/check_track_loading.py` enchaîne `set_map`, `reset(num_cars=1)` et 20 décisions à 20 Hz sur chaque nom. Les 23 circuits se chargent ; seul l'alias échoue, comme décrit plus haut. La version pytest est `uv run pytest -m slow tests/test_tracks.py`. Le chargement par le visualiseur (`viz_replay.py`) n'est pas couvert ici.

## Métriques géométriques : limites

Les champs `geometry_*` sont des **descripteurs géométriques** calculés à partir de la seule centerline. Ce ne sont **pas des mesures de difficulté** : aucun agent n'a encore roulé.

- **Unités :** mètres du **modèle réduit à l'échelle 1:10**, pas du circuit réel. La courbure est en rad/m.
- **Longueur :** périmètre de la centerline fermée.
- **Cap cumulé :** somme des |Δcap| entre segments consécutifs.
- **Courbure en un point :** |Δcap| divisé par la longueur moyenne des deux segments adjacents. Elle est discrète et non lissée, sur un pas d'environ 0,3 m, donc sensible au bruit du tracé. Les maxima (jusqu'à 1,85 rad/m, soit un rayon d'environ 0,54 m) sont inférieurs au rayon de braquage minimal de la voiture (≈ 0,74 m) et reflètent probablement ce bruit plutôt qu'un vrai virage.
- **Virages serrés :** nombre de séquences contiguës de points où la courbure est ≥ 0,5 rad/m (rayon ≤ 2 m).
  - Ce **seuil est arbitraire** : environ 2,7 fois le rayon minimal de braquage.
  - Un même virage bruité peut être compté plusieurs fois.
- **Non pris en compte :**
  - la largeur de piste, fixe à 2,20 m sur tous les circuits (1,1 + 1,1) ;
  - la friction, tirée aléatoirement par le simulateur ;
  - les adversaires ;
  - les chicanes enchaînées ;
  - les zones de freinage.
- **Aucune validation :** rien ne garantit que ces métriques prédisent la performance d'une politique. Elles ne servent ni à rééquilibrer les partitions ni à choisir un curriculum.
