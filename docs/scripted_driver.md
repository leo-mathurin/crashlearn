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
- **Pas de marche arrière près de la ligne** (`progress < no_reverse_below_progress`) : la voiture avance au pas en braquant (voir le bug 2 ci-dessous).

### Contrat d'action

Toute action passe par **une seule borne de sortie** :
- `target_speed` est ramenée dans [−2, 10] et `steering` dans [−0,4189, 0,4189], en marche avant comme en marche arrière ;
- toute valeur non finie devient 0.

Le contrat tient donc pour **n'importe quelle** `DriverConfig`, pas seulement pour les valeurs par défaut. C'est indispensable, puisque les variantes serviront d'adversaires en ligue.

`test_action_contract` le vérifie sur 17 configurations et 12 entrées dégradées :
- **configurations** : vitesses négatives ou supérieures à 10, gains nuls ou forts, fenêtres de 0° ou de 180°, seuils extrêmes, valeurs `nan` ou `inf` ;
- **entrées** : LiDAR `None`, en 2D, vide, NaN ou inf ; adversaire au statut texte ou à la position `nan` ; observations factices d'`agent_loader`.

Deux tests lents vérifient le même contrat dans une vraie course :
- **isolation** : deux pilotes dans la même simulation, puis rejeu de chacun sur une instance neuve, avec des actions identiques ;
- **course dégradée** : 10 rayons NaN par décision, statut d'adversaire en texte et `info` sans `step_count`.

Dans la course dégradée, un rayon NaN est lu comme un obstacle à 0,1 m, par prudence. La voiture se traîne donc (0,017 tour en 300 décisions, contre 0,237 sans dégradation), mais elle avance et ne viole jamais le contrat.

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
| `reverse_speed`, `reverse_steps` | −0,4 m/s, 20 | Marche arrière. **Doit rester au-dessus de −0,5 m/s** (voir le bug 3). |
| `wrong_way_steps` | 5 | Délai avant de lancer un demi-tour. |
| `no_reverse_below_progress` | 0,02 | Pas de marche arrière dans les premiers 2 % d'un tour. |

Les valeurs ont été réglées à la main sur les cartes d'entraînement, sans optimisation automatique. Pour arriver à `max_speed = 7` et `speed_gain = 1,5`, deux configurations ont été essayées sur un tour de chacune des 15 cartes d'entraînement :
- **7 m/s** : les 15 cartes bouclent le tour sans aucun contact ;
- **10 m/s** : 14 cartes sur 15 finissent en DNF.

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

**Seed** : le simulateur n'en expose aucun. Son générateur de bruit (`default_rng(0)`) est créé une seule fois avec le singleton, et `reset()` ne le réinitialise pas. Le script fait donc `close()` avant chaque course, puis remplace ce générateur privé par `default_rng(seed)`. Ce contournement est testé (reproductibilité) et devra être revu après E-12.

## Résultats

Provenance commune :
- commit `1fedea9`, arbre propre (`dirty = false`) ;
- `sim_sha256` `eb6e0732…6ae8`, `f110_gym` 0.2.1, pilote v1 ;
- `max_time` 600 s, friction observée `[1.0]`.

Les données complètes sont dans les JSON produits (`--output`).

Deux protocoles ont été utilisés et **ne sont pas comparables** :
- **P3** : 3 tours, sur les 5 cartes par défaut. C'est le format standard des évaluations internes.
- **P1** : 1 tour, sur les 10 autres cartes d'entraînement. Il ne vérifie que la capacité à boucler un tour.

La course s'arrête dès le tour bouclé. P1 **ne peut donc pas révéler le DNF au passage de ligne** (bug 1), qui ne se déclenche que 80 décisions plus tard.

Dans les tableaux, « Prog. » est la progression cumulée (tours + phase) ; les temps sont en secondes simulées.

### P3, 1 voiture, seeds 0, 1 et 2

| Protocole | Seed | Carte | Statut | Tours | Prog. | Total (s) | Contacts mur | Temps au tour (s) |
|---|---|---|---|---|---|---|---|---|
| P3 | 0 | IMS | DNF | 1 | 1,0916 | – | 0 | 44,35 |
| P3 | 0 | BrandsHatch | DNF | 1 | 1,0737 | – | 0 | 54,95 |
| P3 | 0 | Austin | FINISHED | 3 | 3,0000 | 224,85 | 0 | 75,0 / 74,8 / 75,05 |
| P3 | 0 | MexicoCity | DNF | 2 | 2,0752 | – | 0 | 59,3 / 59,35 |
| P3 | 0 | Budapest | FINISHED | 3 | 3,0000 | 189,10 | 0 | 63,15 / 62,85 / 63,1 |
| P3 | 1 | IMS | DNF | 1 | 1,0922 | – | 0 | 44,3 |
| P3 | 1 | BrandsHatch | DNF | 1 | 1,0741 | – | 0 | 54,9 |
| P3 | 1 | Austin | FINISHED | 3 | 3,0000 | 224,50 | 0 | 75,05 / 74,75 / 74,7 |
| P3 | 1 | MexicoCity | DNF | 1 | 1,0753 | – | 0 | 59,5 |
| P3 | 1 | Budapest | FINISHED | 3 | 3,0000 | 189,20 | 0 | 63,2 / 63,0 / 63,0 |
| P3 | 2 | IMS | DNF | 1 | 1,0919 | – | 0 | 44,3 |
| P3 | 2 | BrandsHatch | DNF | 1 | 1,0740 | – | 0 | 55,0 |
| P3 | 2 | Austin | FINISHED | 3 | 3,0000 | 224,00 | 0 | 74,7 / 74,75 / 74,55 |
| P3 | 2 | MexicoCity | DNF | 1 | 1,0753 | – | 0 | 59,25 |
| P3 | 2 | Budapest | FINISHED | 3 | 3,0000 | 189,05 | 0 | 63,3 / 62,8 / 62,95 |

Deux exécutions avec le seed 0 donnent des JSON identiques, hors date.

### P3, 4 voitures, seed 0

| Protocole | Carte | Voiture | Statut | Tours | Prog. | Total (s) | Contacts mur | Temps au tour (s) |
|---|---|---|---|---|---|---|---|---|
| P3 | Austin | 0 | FINISHED | 3 | 3,0051 | 225,00 | 0 | 75,3 / 74,95 / 74,75 |
| P3 | Austin | 1 | FINISHED | 3 | 3,0000 | 229,95 | 1 | 79,85 / 74,85 / 75,25 |
| P3 | Austin | 2 | FINISHED | 3 | 3,0000 | 226,80 | 0 | 76,85 / 74,9 / 75,05 |
| P3 | Austin | 3 | FINISHED | 3 | 3,0002 | 228,35 | 0 | 78,25 / 74,8 / 75,3 |
| P3 | Budapest | 0 | FINISHED | 3 | 3,0050 | 189,65 | 0 | 63,7 / 62,85 / 63,1 |
| P3 | Budapest | 1 | DNF | 2 | 2,0665 | – | 0 | 65,05 / 62,95 |
| P3 | Budapest | 2 | FINISHED | 3 | 3,0001 | 192,45 | 0 | 66,25 / 63,1 / 63,1 |
| P3 | Budapest | 3 | FINISHED | 3 | 3,0000 | 193,50 | 2 | 67,3 / 63,05 / 63,15 |

Aucune collision entre voitures n'a été relevée.

### P1, 1 voiture, seed 0

| Protocole | Carte | Statut | Tours | Prog. | Temps (s) | Contacts mur | Exposée au bug 1 |
|---|---|---|---|---|---|---|---|
| P1 | Catalunya | FINISHED | 1 | 1,0000 | 65,80 | 0 | oui |
| P1 | Hockenheim | FINISHED | 1 | 1,0000 | 60,85 | 0 | non |
| P1 | Melbourne | FINISHED | 1 | 1,0000 | 75,05 | 0 | non |
| P1 | Monza | FINISHED | 1 | **2,0000** | 73,45 | 0 | oui |
| P1 | Nuerburgring | FINISHED | 1 | 1,0000 | 72,20 | 0 | oui |
| P1 | Sakhir | FINISHED | 1 | 1,0000 | 72,50 | 0 | oui |
| P1 | SaoPaulo | FINISHED | 1 | 1,0000 | 56,80 | 0 | non |
| P1 | Sepang | FINISHED | 1 | **2,0000** | 77,45 | 0 | oui |
| P1 | Silverstone | FINISHED | 1 | 1,0000 | 76,05 | 0 | non |
| P1 | Zandvoort | FINISHED | 1 | 1,0000 | 65,85 | 0 | oui |

Sur Monza et Sepang, la valeur **2,0000** pour un seul tour est un effet du bug 1 : la course s'est arrêtée à la décision exacte où la phase valait 1,0. Sous le protocole P3, ces deux cartes auraient probablement fini en DNF.

### Référence de non-régression

Commande : `--map Budapest --laps 1 --cars 2`.

| Protocole | Voiture | Statut | Temps (s) |
|---|---|---|---|
| 1 tour, 2 voitures | 0 | FINISHED | 63,70 |
| 1 tour, 2 voitures | 1 | FINISHED | 65,40 |

Ces temps sont identiques avant et après les correctifs de revue, et le hash de course aussi (`a1f2643b…`). Ces correctifs n'ont donc pas changé la trajectoire obtenue avec la configuration par défaut.

### Lecture

- **Aucun DNF ne vient du pilote.**
  - Sous P3, aucune voiture seule ne touche un mur, et chaque DNF suit un passage de ligne sur une position exposée au bug 1 : IMS, BrandsHatch, MexicoCity, et la voiture 1 sur Budapest.
  - Austin et Budapest (voiture 0) ne sont pas exposées et bouclent leurs 3 tours.
  - À 4 voitures, deux voitures touchent brièvement le mur (1 et 2 pas en contact) et terminent quand même.
- **Premier tour à plusieurs voitures** : les voitures parties plus loin sur la grille perdent 1 à 4 s, puis roulent aux mêmes temps.
- **P1** : les 10 cartes sont bouclées sans aucun contact. Six d'entre elles sont exposées au bug 1, ce que ce protocole ne permet pas de vérifier.

## Bugs du simulateur révélés (non corrigés, à traiter dans E-12)

1. **DNF systématique après un passage de ligne.**
   - Mécanisme :
     - la centerline est ouverte : 0,36 à 0,46 m séparent son dernier point du premier ;
     - au passage de la ligne, la projection vaut exactement `p_abs = 0.0` ;
     - si la progression de départ de la voiture `p0` vaut environ 1e-19 au lieu de 0, alors `(0.0 - p0) % 1.0` donne exactement `1.0` ;
     - `max_progress` passe donc à 1,0 juste après la remise à zéro du tour, et la voiture ne peut plus le dépasser ;
     - résultat : DNF 80 décisions plus tard, que le pilote soit bon ou non.
   - Traces observées sur IMS : au pas 1310, `rel` vaut 0,0 et le tour est compté. Au pas 1311, `rel` vaut 1,0 et `max_progress` aussi. Le DNF tombe au pas 1391.
   - **Cartes exposées (voiture 0)** : 9 cartes d'entraînement sur 15.
     - Exposées : BrandsHatch, Catalunya, IMS, MexicoCity, Monza, Nuerburgring, Sakhir, Sepang, Zandvoort.
     - Non exposées (`p0 == 0.0`) : Austin, Budapest, Hockenheim, Melbourne, SaoPaulo, Silverstone.
     - Chaque position sur la grille a son propre `p0` : sur Budapest, seule la voiture 1 est exposée.
   - Le déclenchement dépend de la tombée exacte d'une décision sur `p_abs = 0.0`, d'où les variations selon le seed (MexicoCity).
   - **Tout agent RL en est victime** : une évaluation à 3 tours est impossible sur ces cartes tant que ce n'est pas corrigé.
2. **Marche arrière au départ d'un tour.**
   - Même origine : reculer juste après la ligne ramène `rel` vers 0,99, donc `max_progress` vers 1,0, puis DNF au bout de 4 s.
   - Constaté au départ sur IMS : 10 décisions à −1 m/s, puis `max_progress` à 1,0 et DNF au pas 81 alors que la voiture avance.
   - Parade du pilote : `no_reverse_below_progress`.
3. **Divergence du modèle dynamique en marche arrière braquée.**
   - Le moteur passe au modèle dynamique à partir de |v| ≥ 0,5 m/s.
   - En marche arrière à −1,3 m/s, même 0,02 rad de braquage fait exploser la vitesse de lacet (jusqu'à 1e32 rad/s) et le cap devient aléatoire.
   - En ligne droite, ou sous 0,5 m/s (modèle cinématique), le comportement reste stable. La voiture se rétablit en repartant droit en avant.
   - Parade du pilote : `reverse_speed = -0.4`.
   - **Un agent RL qui recule vite en braquant exploitera ou subira ce bug.**
4. **Friction fixe à 1** : `_update_friction` ne modifie pas la physique. Toutes les mesures ci-dessus sont faites à friction constante (`friction_seen = [1.0]`) et ne disent rien du comportement sous pluie.
5. **Pas de seed public**, et `reset()` ne réinitialise pas le générateur (voir « Commandes »).

## Limites connues

- **Cartes mesurées sur 3 tours** : IMS, BrandsHatch, Austin, MexicoCity et Budapest.
  - Les 10 autres cartes d'entraînement n'ont été mesurées que sous P1 (1 tour), qui ne peut pas révéler le bug 1. Six d'entre elles y sont exposées.
  - Validation et test scellé sont exclus volontairement : le test n'est ouvert qu'une fois, après le gel.
- **Récupération limitée.**
  - Elle fonctionne après un contact simple : sur Austin à 10 m/s, la voiture recule puis reprend sa progression.
  - Après un tête-à-queue qui laisse la voiture en travers de la piste (2,2 m de large), le demi-tour en plusieurs manœuvres dépasse les 4 s de la règle de stagnation, puisqu'il faut reculer à moins de 0,5 m/s. La voiture finit en DNF.
  - Aux valeurs par défaut, ce cas ne s'est pas produit.
- **Pas de notion de sens de course** : le gap follower peut faire demi-tour après un contact. Seule la détection par `progress` le corrige.
- **Temps au tour non optimisés** : trajectoire par le centre libre et plafond à 7 m/s. Ce pilote est une référence basse, pas un objectif.
- **Résolution angulaire** : environ 3,6° par rayon, ce qui rend le pilote moins précis dans les chicanes serrées.
