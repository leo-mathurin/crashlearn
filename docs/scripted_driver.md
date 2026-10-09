# Pilote scripté de contrôle (E-11)

> **Témoin de diagnostic, pas un livrable.** Ce pilote n'apprend rien. Il ne va pas dans `submission/` et ne produit pas d'ONNX. Le sujet exige des politiques apprises : les livrables restent les trois pilotes RL et le champion.

Ce pilote sert à deux choses :
- **Distinguer les sources d'erreur.** Quand une métrique ou le moteur a un comportement étrange, on relance la même course avec ce pilote déterministe. On sait alors si le problème vient du simulateur, de la mesure ou de la politique RL.
- **Servir d'adversaire simple** dans la ligue du champion. L'interface est prête ; la ligue elle-même reste à construire.

Code : [`src/crashlearn/scripted_driver.py`](../src/crashlearn/scripted_driver.py) (NumPy seul) et [`scripts/control_race.py`](../scripts/control_race.py).

## Stratégie retenue : gap follower LiDAR

Le pilote reçoit le LiDAR à 100 rayons et choisit sa direction à chaque décision :
1. Il ne garde que les rayons situés à ±100° de l'axe.
2. Il élargit chaque obstacle d'une demi-largeur de voiture plus une marge (*disparity extender*).
3. Parmi les rayons les plus profonds, il vise celui qui est le plus proche de l'axe.

Braquage et vitesse :
- **braquage** = `steer_gain × angle` de ce rayon ;
- **vitesse** : elle dépend de la distance libre devant (±10°) et diminue quand le braquage augmente.

Pourquoi pas le suivi de trajectoire :
- **l'observation ne contient pas la pose du véhicule**. Suivre la centerline ou la raceline imposerait de lire l'état privé du moteur, ce qui dépasse l'interface imposée aux agents ;
- Montreal et Shanghai n'ont pas de raceline.

Le gap follower n'utilise que l'observation publique et fonctionne sur toutes les cartes. En contrepartie, il ne suit pas une trajectoire optimale et n'a aucune notion du sens de la course (voir « Récupération »). Implémenter les deux approches doublerait la maintenance sans apporter de meilleur témoin.

**Même interface que le tournoi** : `ScriptedDriver().predict(obs, info) -> (target_speed, steering)`, qui renvoie des floats Python finis et bornés. Le pilote se branche donc sans adaptateur dans les courses de contrôle et dans la ligue.

Un nouvel épisode est détecté quand `info["step_count"] == 0`. C'est la valeur du simulateur au premier appel après `reset()`, vérifiée par un test. Une clé absente **n'est pas** un nouvel épisode : l'état est conservé et la récupération continue de fonctionner. Tout l'état est porté par l'instance : plusieurs pilotes dans la même simulation ne partagent rien, ce qui est testé.

**Adversaires** : le LiDAR ne voit que les murs. Chaque adversaire actif (`opponents[slot]["status"] == 1`) est donc ajouté au scan comme un obstacle de 0,35 m de rayon.

**Déterminisme** : le pilote n'utilise aucune source d'aléa. Avec la même carte, le même seed et la même configuration, deux exécutions produisent la même trajectoire, contrôlée par `race_trajectory_sha256`.

### Récupération

Trois mécanismes gèrent les situations bloquantes :
- **Contact mur ou voiture immobile** pendant `stuck_steps` décisions : marche arrière pendant `reverse_steps` décisions, en braquant à l'opposé, pour réorienter le nez.
- **Mauvais sens** (`progress` décroît pendant `wrong_way_steps` décisions alors que la voiture avance) : demi-tour à braquage maximal, du côté de l'espace arrière le plus dégagé.
- **Pas de marche arrière près de la ligne** (`progress < no_reverse_below_progress`) : la voiture avance au pas en braquant, pour ne pas faire reculer son maximum de progression juste après le passage de ligne.

### Contrat d'action

Toute action passe par **une seule borne de sortie** :
- `target_speed` est ramenée dans [−2, 10] et `steering` dans [−0,4189, 0,4189], en marche avant comme en marche arrière ;
- toute valeur non finie devient 0.

Le contrat tient donc pour **n'importe quelle** `DriverConfig`, pas seulement pour les valeurs par défaut. C'est indispensable, puisque les variantes serviront d'adversaires en ligue.

`test_action_contract` le vérifie sur 17 configurations et 12 entrées dégradées :
- **configurations** : vitesses négatives ou supérieures à 10, gains nuls ou forts, fenêtres de 0° ou de 180°, seuils extrêmes, valeurs `nan` ou `inf` ;
- **entrées** : LiDAR `None`, en 2D, vide, NaN ou inf ; adversaire au statut texte ou à la position `nan` ; observations factices d'`agent_loader`.

Trois tests lents vérifient le même contrat dans une vraie course :
- **isolation** : deux pilotes dans la même simulation, puis rejeu de chacun sur une instance neuve, avec des actions identiques ;
- **course dégradée** : 10 rayons NaN par décision, statut d'adversaire en texte et `info` sans `step_count` ;
- **récupération réelle** : le pilote est aveuglé par un LiDAR à 15 m pendant 60 décisions, donc il entre dans le mur, puis retrouve le vrai LiDAR. Il doit repartir et dépasser d'au moins 0,1 tour la progression du contact. Aux valeurs par défaut, c'est le seul moyen d'exercer la marche arrière : le pilote ne touche aucun mur en course normale.

Dans la course dégradée, un rayon NaN est lu comme un obstacle à 0,1 m, par prudence. La voiture se traîne donc, et elle peut finir en DNF pour stagnation autour de la décision 270 (fenêtre de 200 pas, gain minimal de 1 m). C'est un résultat, pas une violation du contrat : le test exige que le DNF vienne bien de la stagnation et non d'une sortie de piste.

**Formes non supportées** : `opponents` en liste, `agent_id` à `None`, LiDAR en texte. Ces formes n'existent pas dans le simulateur. Leur prise en charge attendra que la ligue accepte des observations de sources externes.

## Paramètres (`DriverConfig`, version `1`)

Toutes les valeurs sont regroupées dans un dataclass figé et recopiées en entier dans chaque enregistrement de course. Toute modification d'une valeur par défaut doit s'accompagner d'une nouvelle `version`.

| Paramètre | Défaut | Effet |
|---|---|---|
| `fov_deg` | 100 | Demi-angle de la fenêtre où l'on choisit la direction. Plus large : réagit plus tôt aux virages, mais risque de viser un côté. |
| `disparity_threshold` | 0,5 m | Écart entre deux rayons voisins à partir duquel on considère un bord d'obstacle. |
| `car_half_width`, `margin` | 0,16 m, 0,25 m | Largeur de l'élargissement des obstacles. Plus grand : passe plus loin des murs et coupe moins les virages. |
| `target_tolerance` | 0,5 m | Rayons retenus comme candidats (proches du plus profond). Plus grand : trajectoire plus droite. |
| `steer_gain` | 1,0 | Braquage par radian d'angle visé. |
| `front_cone_deg`, `front_margin` | 10°, 1 m | Mesure de la distance libre devant, et distance à laquelle la vitesse tombe à `min_speed`. |
| `speed_gain`, `min_speed`, `max_speed` | 1,5, 1, 7 m/s | Vitesse = `min_speed + speed_gain × (distance libre − front_margin)`, plafonnée à `max_speed`. |
| `steer_slowdown` | 0,5 | Part de la vitesse perdue au braquage maximal. |
| `stuck_speed`, `stuck_steps` | 0,2 m/s, 10 | Seuil de vitesse et durée pour considérer la voiture bloquée. |
| `reverse_speed`, `reverse_steps` | −0,4 m/s, 20 | Marche arrière. **Garder au-dessus de −0,5 m/s**, seuil sous lequel le moteur reste cinématique : reculer plus vite en braquant faisait diverger l'ancienne archive, et la manœuvre lente suffit. |
| `wrong_way_steps` | 5 | Délai avant de lancer un demi-tour. |
| `no_reverse_below_progress` | 0,02 | Pas de marche arrière dans les premiers 2 % d'un tour. |

Les valeurs ont été réglées à la main sur les cartes d'entraînement, sans optimisation automatique. Pour arriver à `max_speed = 7` et `speed_gain = 1,5`, deux configurations ont été essayées sur un tour de chacune des 15 cartes d'entraînement :
- **7 m/s** : les 15 cartes bouclent le tour, sans aucun contact mur ;
- **10 m/s** : 2 cartes sur 15 seulement, 13 DNF et 14 cartes avec contact mur.

Mesure refaite sur l'archive 2026-09-18.

## Commandes

```bash
uv run python scripts/control_race.py                      # 5 cartes train, 3 tours, seed 0, 1 voiture
uv run python scripts/control_race.py --map Austin --cars 4 --seed 1 --output runs/austin.json
uv run pytest tests/test_scripted_driver.py                # contrat d'action, déterminisme, CLI (CI)
uv run pytest -m slow tests/test_scripted_driver.py        # courses réelles : reproductibilité, isolation, entrées dégradées
```

Options du script :
- `--map` : répétable. Uniquement des cartes **d'entraînement** canoniques ; validation, test scellé et alias sont refusés avec le code 2.
- `--seed` : bruit LiDAR.
- `--laps` : 1 à 3, **3 par défaut**. C'est le format des évaluations internes, et le simulateur fige la voiture en FINISHED au 3e tour. On n'hérite pas du `--laps 2` de `sim_recorder.py`.
- `--cars` : 1 à 4 voitures, chacune avec son pilote.
- `--max-time` : secondes simulées avant `TIMEOUT` (600 par défaut). La valeur doit être finie et d'au moins 0,05 s (une décision), sinon le script sort avec le code 2.
- `--output` : chemin de sortie (`runs/control_race.json` par défaut, ignoré par git).
  - L'accès en écriture est vérifié **avant la première course** ; en cas d'échec, sortie avec le code 2.
  - Le JSON complet est réécrit après chaque course : un plantage en cours de campagne garde les courses terminées.

Chaque voiture produit un enregistrement JSON avec :
- **le résultat** : statut (`FINISHED`/`DNF`/`TIMEOUT`), tours, progression cumulée, temps au tour et total, décisions, pas en contact mur ou véhicule, distance parcourue, `dnf_step`, friction observée, `race_trajectory_sha256` ;
- **la provenance** : commit et état dirty, `sim_sha256` (hash de `env_simulation.py`), version de `f110_gym`, seed, voitures, tours, `max_time_s`, `DriverConfig` complète et date.

Précisions sur ces champs :
- **`race_trajectory_sha256`** : il couvre les actions et la progression de **toutes** les voitures de la course, donc il est identique dans chacun de ses enregistrements. Il sert à vérifier qu'une exécution en reproduit une autre.
  - L'ancien nom `trajectory_sha256` n'apparaît dans aucun résultat conservé. Les mesures ci-dessous ont été régénérées avec le nouveau nom, sans version de schéma.
- **`dirty`** : il ne tient compte que des fichiers suivis modifiés. Un dossier local non suivi comme `.claude/` ne le rend pas vrai. Il vaut `null` si git est indisponible.

Le script sort avec le code 0 même en cas de DNF : un échec est une mesure.

**Seed** : `reset(num_cars, seed=)` de l'archive 2026-09-18 réarme le bruit du wrapper et le moteur. Le script l'appelle avant chaque course, sans aucun accès à l'état privé du simulateur. Deux tests le couvrent : même seed, mêmes JSON ; seed différent, hash de trajectoire différent.

## Résultats

Provenance commune :
- commit `9c290bc`, arbre propre (`dirty = false`) ;
- `sim_sha256` `3f4aac8e…92cb` (archive 2026-09-18), `f110_gym` 0.2.1, pilote v1 ;
- `max_time` 600 s, friction observée `[1.0]`.

Les données complètes sont dans les JSON produits (`--output`).

Deux protocoles, qui **ne sont pas comparables** :
- **P3** : 3 tours, sur les 5 cartes par défaut. C'est le format standard des évaluations internes.
- **P1** : 1 tour, sur les 10 autres cartes d'entraînement. Il ne vérifie que la capacité à boucler un tour.

Dans les tableaux, « Prog. » est la progression cumulée (tours + phase) ; les temps sont en secondes simulées.

### P3, 1 voiture, seeds 0, 1 et 2

Aucun DNF, aucun contact mur, aucune collision entre voitures.

| Carte | Statut | Total seed 0 | Total seed 1 | Total seed 2 | Temps au tour, seed 0 |
|---|---|---|---|---|---|
| IMS | FINISHED | 132,35 | 132,20 | 132,20 | 44,3 / 44,1 / 43,95 |
| BrandsHatch | FINISHED | 163,50 | 163,40 | 163,80 | 54,85 / 54,3 / 54,35 |
| MexicoCity | FINISHED | 176,80 | 177,30 | 177,10 | 59,2 / 58,8 / 58,8 |
| Budapest | FINISHED | 189,45 | 189,20 | 189,30 | 63,35 / 63,05 / 63,05 |
| Austin | FINISHED | 224,30 | 223,75 | 223,75 | 75,1 / 74,95 / 74,25 |

L'écart entre seeds reste sous 0,6 s sur 3 tours. Deux exécutions avec le même seed donnent des JSON identiques, hors date.

### P3, 4 voitures, seed 0

| Carte | Voiture | Statut | Total | Temps au tour |
|---|---|---|---|---|
| Austin | 0 | FINISHED | 224,85 | 74,85 / 74,95 / 75,05 |
| Austin | 1 | FINISHED | 227,75 | 77,85 / 75,3 / 74,6 |
| Austin | 2 | FINISHED | 226,00 | 76,25 / 75,15 / 74,6 |
| Austin | 3 | FINISHED | 229,15 | 79,2 / 75,35 / 74,6 |
| Budapest | 0 | FINISHED | 190,35 | 64,5 / 63,05 / 62,8 |
| Budapest | 1 | FINISHED | 189,25 | 63,45 / 63,1 / 62,7 |
| Budapest | 2 | FINISHED | 191,90 | 66,0 / 62,75 / 63,15 |
| Budapest | 3 | FINISHED | 193,40 | 67,0 / 63,1 / 63,3 |

Les quatre voitures terminent, sans contact mur ni contact entre elles. Les places arrière de la grille perdent 1 à 4 s au premier tour, puis roulent aux mêmes temps.

### P1, 10 autres cartes d'entraînement, seed 0

Les 10 cartes bouclent leur tour, sans contact mur.

| Carte | Temps (s) | Carte | Temps (s) |
|---|---|---|---|
| SaoPaulo | 56,90 | Nuerburgring | 72,20 |
| Hockenheim | 60,40 | Sakhir | 72,40 |
| Catalunya | 65,80 | Melbourne | 75,00 |
| Zandvoort | 65,80 | Silverstone | 76,20 |
| Monza | 71,30 | Sepang | 77,50 |

Le test lent `test_one_lap_on_every_training_track` rejoue ce protocole sur les **15** cartes d'entraînement et échoue au moindre DNF ou contact mur (environ 20 s).

### Référence de reproductibilité

Commande : `--map Budapest --laps 1 --cars 2`.

| Voiture | Statut | Temps (s) |
|---|---|---|
| 0 | FINISHED | 64,65 |
| 1 | FINISHED | 63,45 |

### Lecture

- **Le pilote boucle les 15 cartes d'entraînement**, à 1 comme à 3 tours, seul ou à 4 voitures, sans jamais toucher un mur aux valeurs par défaut.
- **Les DNF mesurés sur l'archive 2026-09-08 ont disparu.** Ils venaient du faux DNF après le passage de ligne, corrigé en amont : à configuration de pilote inchangée, IMS, BrandsHatch et MexicoCity terminent désormais leurs 3 tours. C'est la confirmation du diagnostic, et l'intérêt d'un témoin déterministe.
- **Les temps au tour sont stables** : moins de 1 s d'écart entre les tours d'une même course, et moins de 0,6 s entre seeds.

## Ce que ce pilote a révélé dans le simulateur

Les défauts trouvés avec ce témoin sur la première archive, et leur devenir, sont consignés dans [docs/audit_simulateur.md](audit_simulateur.md) :
- **faux DNF après le passage de ligne** (la progression de départ valait environ 1e-19 au lieu de 0, donc le maximum de progression passait à 1,0) : corrigé en amont ;
- **divergence du modèle dynamique en marche arrière braquée** au-delà de 0,5 m/s (vitesse de lacet jusqu'à 1e32 rad/s) : corrigée en amont ;
- **absence de graine** : `reset` en prend une désormais.

Reste ouvert : la **friction est figée à 1,0** (`_update_friction` ne l'applique pas). Toutes les mesures ci-dessus valent donc à friction constante et ne disent rien du comportement sous pluie. La friction variable est traitée côté entraînement (E-23).

## Limites connues

- **Périmètre mesuré** : 5 cartes sur 3 tours, 10 cartes sur 1 tour, seeds 0 à 2. Validation et test scellé sont exclus volontairement : le test n'est ouvert qu'une fois, après le gel.
- **Récupération peu exercée** : aux valeurs par défaut, le pilote ne touche aucun mur. Seul le test lent qui l'aveugle volontairement en exerce la marche arrière.
  - Une fois en travers d'une piste large d'environ 2,2 m, le demi-tour reste lent, puisqu'il faut reculer à moins de 0,5 m/s pour rester dans le modèle cinématique. Il peut donc dépasser la fenêtre de stagnation (200 pas, soit 10 s, avec 1 m de gain minimal) et finir en DNF.
- **Pas de notion de sens de course** : le gap follower peut faire demi-tour après un contact. Seule la détection par `progress` le corrige.
- **Temps au tour non optimisés** : trajectoire par le centre libre et plafond à 7 m/s. Ce pilote est une référence basse, pas un objectif.
- **Résolution angulaire** : environ 3,6° par rayon, ce qui rend le pilote moins précis dans les chicanes serrées.
- **Friction figée à 1,0** : voir ci-dessus.
