# Dashboard de course

Lecteur web local de replays de course, dans une scène 3D. Il lit les fichiers `.npz` de `vendor/simulation/sim_recorder.py`, sans compte, sans base distante et sans streaming. Vocabulaire et périmètre : [Dashboard de course et vocabulaire](https://linear.app/e-hamilton/document/dashboard-de-course-et-vocabulaire-0987cb2ff84a).

## Démarrer

Prérequis : [Bun](https://bun.sh) 1.4 ou plus récent, et `uv` pour enregistrer des replays.

```bash
uv run python scripts/record_replay.py --demo    # une fois, depuis la racine : 3 replays dans replays/
cd dashboard && bun install && bun start         # puis ouvrir http://localhost:5173
```

La page liste les replays du dossier `replays/` (ignoré par Git). On peut aussi choisir un fichier `.npz` ou le déposer dans la fenêtre. Un replay ouvert depuis la liste a une adresse stable : `http://localhost:5173/?replay=demo-4-voitures-spielberg.npz`.

Options du serveur : `bun start --port 8080 --replays <dossier> --maps <dossier>`. `bun dev` recharge la page à chaque modification du code.

### Enregistrer un replay

`scripts/record_replay.py` lance le `sim_recorder.py` fourni, sans le modifier, avec les pilotes de `submission/`. Il refuse les circuits du test scellé.

```bash
uv run python scripts/record_replay.py --map Sochi --drivers driver1 driver2 --seed 4
uv run python scripts/record_replay.py --map Austin --drivers driver1 driver2 driver1 crash-test
```

`crash-test` n'est pas un pilote livré : cette voiture roule tout droit à 3 m/s et finit en DNF. Elle sert à enregistrer un vrai abandon. Le replay `demo-dnf-sochi` en contient une.

## Utiliser

| Action                               | Souris ou clavier                           |
| ------------------------------------ | ------------------------------------------- |
| Lecture, pause                       | bouton ou `Espace`                          |
| Avancer ou reculer d'une frame       | `→` `←` (20 frames, soit 1 s, avec `Maj`)   |
| Début, fin                           | `Début`, `Fin`                              |
| Vitesse de lecture, de 0,25× à 8×    | boutons ou `↑` `↓`                          |
| Zoom                                 | molette ou pincement, centré sur le curseur |
| Déplacement                          | glisser                                     |
| Revenir à la vue d'ensemble          | double-clic ou `R`                          |
| Afficher ou masquer la limite de DNF | interrupteur ou `L`                         |
| Ouvrir un replay                     | bouton ou `O`                               |
| Afficher les mesures de fluidité     | `P`                                         |

Vu de loin, une voiture de 0,58 m fait moins d'un pixel : une pastille de couleur marque alors sa position. Elle disparaît quand on zoome assez pour voir la voiture.

## Ce que la scène montre

Tout ce qui est sur la piste vient des données du simulateur. Rien n'est inventé.

| Élément                      | Source                                                                                                                                                |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bitume                       | centerline et demi-largeurs de `<Carte>_centerline.csv`                                                                                               |
| Lignes de bord, vibreurs     | même bord de piste ; rouge et blanc là où la centerline tourne à plus de 0,14 rad/m. Les vibreurs sont de la peinture sur le bord mesuré, sans relief |
| Ligne de départ              | premier point de la centerline, la pose de départ du simulateur                                                                                       |
| Murs, 0,25 m de haut         | pixels occupés de `<Carte>_map.png`, au seuil du simulateur (valeur ≤ 128). Un rayon LiDAR s'arrête sur ces mêmes pixels                              |
| Limite de DNF, tirets rouges | bord de piste + 1 m (`_OFFTRACK_MARGIN_M`), avec le calcul de `_is_off_track`                                                                         |
| Voitures                     | poses enregistrées, à la taille de la boîte de collision : 0,58 × 0,31 m                                                                              |

| LiDAR de la Voiture sélectionnée | les 100 distances enregistrées, avec les angles du simulateur (`_LIDAR_SCAN_FOV`). Un point marque chaque rayon qui a touché un mur ; un rayon à 15 m n'a rien touché |
| Marqueur « DNF » | dernière pose d'une voiture en abandon, à partir de la frame où le simulateur déclare le DNF. Pas de marqueur pour une arrivée |

Une Voiture inactive (statut DNF ou arrivée) est masquée : le simulateur la retire de la piste. Entre deux frames (20 Hz), la pose est interpolée pour un mouvement fluide ; les valeurs affichées sont celles de la frame enregistrée.

Entre deux frames, le bout de chaque rayon LiDAR glisse en ligne droite de son point à la frame précédente à son point à la frame suivante. En pause sur une frame, ce sont exactement les distances enregistrées.

## Caméras et Mini-carte

Une seule caméra est à l'écran à la fois. Changer de caméra ne change pas la position de lecture.

| Caméra    | Vue                                                                                                                                                     |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Aérienne  | orthographique, nord en haut ; c'est la caméra à l'ouverture                                                                                            |
| Libre     | perspective en orbite autour d'un point du sol. À la première utilisation, elle cadre ce que montrait la caméra précédente, puis elle garde sa position |
| Poursuite | derrière la Voiture suivie, tournée vers l'avant                                                                                                        |

- **Voiture suivie** : c'est la Voiture sélectionnée par défaut. Le menu « Suivre » choisit une autre voiture : elle est alors déliée, et la sélection peut changer sans déplacer la caméra. « Voiture sélectionnée » les lie de nouveau.
- **Voiture suivie inactive** : quand elle devient inactive, la caméra reste 2 s sur place (temps réel), puis suit la voiture active la mieux classée. Si les deux voitures sont liées, la Voiture sélectionnée change avec elle. Une voiture déjà inactive que l'on choisit garde la caméra : on peut regarder l'endroit de son DNF.
- **Transitions** : un clic fait glisser la vue d'une caméra à l'autre en 0,65 s. Au clavier, le changement est immédiat. Avec « réduire les animations » dans le système, il l'est toujours.
- **Mini-carte** : toujours visible, affichage seul. Le circuit est une image de la scène 3D prise d'en haut, avec le rendu de la caméra Aérienne ; elle est refaite quand la limite de DNF change. Par-dessus : un point par voiture active (cerclé de blanc pour la Voiture sélectionnée), une croix rouge par marqueur DNF, et un cercle à l'endroit que regarde la caméra.

## Classement et télémétrie

- **Classement** : ordre du simulateur (`ranks`), nom, couleur, tour et progression du tour. Une voiture inactive affiche « Arrivée » ou « DNF » avec le temps simulé de sa fin de course. À la dernière frame, c'est le classement final. Les marques de couleur sur la timeline donnent ces mêmes instants.
- **Télémétrie de la Voiture sélectionnée** : vitesse mesurée et vitesse cible, braquage mesuré et braquage demandé (le trait blanc de chaque jauge est la consigne du pilote), tour, adhérence de la piste, et courbe des deux vitesses sur 6 s avant et après la tête de lecture.
- **Données absentes** : un ancien replay peut ne pas avoir `ranks`, `lidar`, `velocity` ou `actions`. La valeur est alors notée « indisponible », jamais remplacée par zéro. Sans `ranks`, l'ordre est celui de `viz_replay.py` : arrivées, puis voitures actives par distance, puis DNF.

Le recorder ne garde ni le nombre de tours visé ni la raison d'un DNF, et il coupe les noms à 10 caractères : le dashboard affiche ce que le fichier contient. Ces enrichissements sont le sujet de E-35.

Si la carte du replay est introuvable, les voitures sont affichées sur une grille neutre, avec un avertissement. Si le fichier n'est pas un replay valide, la page dit pourquoi et reste utilisable.

## Technique

| Choix                                         | Raison                                                                                                                                                   |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Three.js, `WebGPURenderer`                    | un seul code de scène : WebGPU quand le navigateur l'offre, WebGL 2 sinon (Firefox et Chrome sous Linux aujourd'hui)                                     |
| TypeScript strict, sans framework d'interface | le HUD est mis à jour à chaque frame : du DOM direct coûte moins qu'un rendu virtuel, et il n'y a que deux dépendances à l'exécution (`three`, `fflate`) |
| Bun pour tout : serveur, bundler, tests       | `Bun.serve` sert la page compilée à la volée, les replays et les cartes de `vendor/` sans copie. Pas de Vite, pas de Node                                |
| Décodage NPZ dans le navigateur               | `fflate` pour le ZIP, lecteur `.npy` maison : le format du recorder reste la seule source                                                                |
| Textures procédurales                         | bitume, herbe et béton sont dessinés au chargement avec une graine fixe : aucun fichier image, même rendu partout                                        |
| oxlint, oxfmt, `tsc`, `bun test`, Playwright  | mêmes outils que les autres projets de l'équipe                                                                                                          |

```
server.ts            serveur local : page, /api/replays, /api/maps
src/replay/          décodage NPZ, validation, horloge de lecture, classement, valeurs par voiture, Voiture suivie
src/track/           données de piste : centerline, limite de DNF, murs (sans Three.js, testé)
src/scene/           rendu : renderer, lumières, circuit, voitures, LiDAR, marqueur DNF, caméras
src/ui/              HUD : timeline, classement, télémétrie, caméras, Mini-carte, étiquettes, choix du replay, mesures
tests/               tests unitaires (bun test)
e2e/                 tests navigateur, Chromium et Firefox (Playwright)
```

## Qualité

Ce sont les commandes de la CI (`.github/workflows/dashboard.yml`).

```bash
bun run lint && bun run format:check && bun run typecheck && bun run test && bun run build
```

Tests navigateur, hors CI : ils enregistrent les replays de démonstration s'ils manquent.

```bash
bunx playwright install chromium firefox    # une fois
bun run test:e2e                            # sur une machine sans écran : xvfb-run -a bun run test:e2e
```

Sans écran, Firefox n'a pas de WebGL : `xvfb-run` lui en donne un.

`window.dashboard` donne l'état de la page aux tests (`state()`) et mesure le coût d'une frame sans la limite de l'écran (`benchmark()`).
